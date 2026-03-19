#!/usr/bin/env python3
"""Closed-loop validation for robust LQI gains.

This script is intentionally separated from tune_lqi_robust.py so you can:
- test a manually edited gain matrix,
- or validate gains loaded from a prior lqi_tuned.yaml,
without running synthesis again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from tune_lqi_robust import (  # reusing common model/simulation helpers
    AXIS_TO_INDEX,
    INPUT_NAMES,
    STATE_NAMES,
    Scenario,
    _as_bool,
    _as_float,
    _build_augmented_matrices,
    _build_vertices,
    _evaluate_linear_stability,
    _load_model_gammas,
    _plot_case,
    _scenario_to_ref_amp,
    _simulate_closed_loop,
    _write_rows_csv,
)


def _resolve_path(path_raw: str, cfg_dir: Path) -> Path:
    p = Path(path_raw)
    if not p.is_absolute():
        p = (cfg_dir / p).resolve()
    return p


def _load_runtime_gain_and_limits(
    cfg: Dict,
    cfg_dir: Path,
    tuned_yaml_override: Optional[Path],
) -> Dict:
    ctrl_cfg = cfg.get("controller", {})

    tuned_data = None
    tuned_src = None
    if tuned_yaml_override is not None:
        tuned_src = tuned_yaml_override.resolve()
    elif ctrl_cfg.get("source_tuned_yaml"):
        tuned_src = _resolve_path(str(ctrl_cfg.get("source_tuned_yaml")), cfg_dir)

    if tuned_src is not None:
        tuned_data = yaml.safe_load(tuned_src.read_text()) or {}

    tuned_controller = (tuned_data or {}).get("controller", {})

    control_sign = _as_float(
        ctrl_cfg.get("control_sign", tuned_controller.get("control_sign", -1.0)),
        -1.0,
    )

    integral_limits = np.asarray(
        ctrl_cfg.get("integral_limits", tuned_controller.get("integral_limits", [0.8, 0.8, 0.8, 1.2])),
        dtype=float,
    )
    if integral_limits.shape != (4,):
        raise ValueError("controller.integral_limits must have 4 elements")

    v_limits = np.asarray(
        ctrl_cfg.get("v_limits", tuned_controller.get("v_limits", [0.5, 0.5, 0.5, 0.5])),
        dtype=float,
    )
    if v_limits.shape != (4,):
        raise ValueError("controller.v_limits must have 4 elements")

    k_flat = ctrl_cfg.get("k_runtime_row_major")
    if k_flat is None:
        k_flat = ctrl_cfg.get("k")
    if k_flat is None:
        k_flat = tuned_controller.get("k_runtime_row_major")

    if k_flat is None:
        raise ValueError(
            "Missing runtime gain matrix. Provide controller.k_runtime_row_major (or controller.k) "
            "or set controller.source_tuned_yaml / --tuned-yaml"
        )

    k_arr = np.asarray(k_flat, dtype=float)
    if k_arr.size != 48:
        raise ValueError("Runtime K must have exactly 48 elements (row-major 4x12)")
    k_runtime = k_arr.reshape(4, 12)

    return {
        "control_sign": float(control_sign),
        "integral_limits": integral_limits,
        "v_limits": v_limits,
        "k_runtime": k_runtime,
        "k_source": str(tuned_src) if tuned_src is not None else "config",
    }


def run(config_path: Path, tuned_yaml_override: Optional[Path], out_dir_override: Optional[Path]) -> None:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg_dir = config_path.parent

    g_nom = _load_model_gammas(cfg, cfg_dir)

    model_cfg = cfg.get("model", {})
    uncertainty_pct = _as_float(model_cfg.get("uncertainty_pct"), 0.20)
    vertices = _build_vertices(g_nom, uncertainty_pct)

    a_vertices: List[np.ndarray] = []
    b_vertices: List[np.ndarray] = []
    for gv in vertices:
        a, b = _build_augmented_matrices(gv, g_nom)
        a_vertices.append(a)
        b_vertices.append(b)

    rt = _load_runtime_gain_and_limits(cfg, cfg_dir, tuned_yaml_override)
    control_sign = rt["control_sign"]
    integral_limits = rt["integral_limits"]
    v_limits = rt["v_limits"]
    k_runtime = rt["k_runtime"]

    sim_cfg = cfg.get("simulation", {})
    dt = _as_float(sim_cfg.get("dt"), 0.02)
    duration = _as_float(sim_cfg.get("duration"), 8.0)
    settling_band = _as_float(sim_cfg.get("settling_band"), 0.02)

    raw_scenarios = sim_cfg.get(
        "scenarios",
        [
            {"name": "step_x", "axis": "x", "amplitude": 0.2, "step_time": 0.5},
            {"name": "step_y", "axis": "y", "amplitude": 0.2, "step_time": 0.5},
            {"name": "step_z", "axis": "z", "amplitude": 0.2, "step_time": 0.5},
            {"name": "step_yaw", "axis": "yaw", "amplitude": 10.0, "unit": "deg", "step_time": 0.5},
        ],
    )
    scenarios: List[Scenario] = []
    for s_raw in raw_scenarios:
        axis = str(s_raw.get("axis", "")).strip().lower()
        if axis not in AXIS_TO_INDEX:
            raise ValueError(f"Invalid scenario axis '{axis}'")
        amp = _scenario_to_ref_amp(axis, float(s_raw.get("amplitude", 0.2)), str(s_raw.get("unit", "rad")))
        scenarios.append(
            Scenario(
                name=str(s_raw.get("name", f"step_{axis}")),
                axis=axis,
                amplitude=amp,
                step_time=float(s_raw.get("step_time", 0.5)),
            )
        )

    val_cfg = cfg.get("validation", {})
    simulate_worst_vertex = _as_bool(val_cfg.get("simulate_worst_vertex"), True)

    out_cfg = cfg.get("output", {})
    if out_dir_override is not None:
        out_dir = out_dir_override.resolve()
    else:
        out_dir_raw = out_cfg.get("dir", "src/tello_control_pkg/experiments_lqi/validation_latest")
        out_dir = Path(out_dir_raw)
        if not out_dir.is_absolute():
            out_dir = (cfg_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_plots = _as_bool(out_cfg.get("save_plots"), True)
    save_csv = _as_bool(out_cfg.get("save_csv"), True)

    stability = _evaluate_linear_stability(a_vertices, b_vertices, k_runtime, control_sign, dt)

    worst_idx = 0
    worst_real = -1e9
    for i, (a, b) in enumerate(zip(a_vertices, b_vertices)):
        acl = a + b @ (control_sign * k_runtime)
        rm = float(np.max(np.real(np.linalg.eigvals(acl))))
        if rm > worst_real:
            worst_real = rm
            worst_idx = i

    cases = [("nominal", g_nom)]
    if simulate_worst_vertex:
        cases.append(("worst_vertex", vertices[worst_idx]))

    metrics_all: Dict[str, Dict[str, float]] = {}
    for case_name, g_case in cases:
        for sc in scenarios:
            m, rows = _simulate_closed_loop(
                scenario=sc,
                g_plant=g_case,
                g_nom=g_nom,
                k_runtime=k_runtime,
                control_sign=control_sign,
                dt=dt,
                duration=duration,
                integral_limits=integral_limits,
                v_limits=v_limits,
                settling_band=settling_band,
            )
            key = f"{case_name}:{sc.name}"
            metrics_all[key] = m

            if save_csv:
                _write_rows_csv(rows, out_dir / "csv" / f"{key}.csv")
            if save_plots:
                _plot_case(rows, title=key, png_path=out_dir / "plots" / f"{key}.png")

    k_index_map = []
    for r_idx, u_name in enumerate(INPUT_NAMES):
        for c_idx, st_name in enumerate(STATE_NAMES):
            flat_idx = r_idx * 12 + c_idx
            k_index_map.append(f"k[{flat_idx:02d}] = K[{r_idx},{c_idx}] -> {u_name} <- {st_name}")

    result = {
        "config": str(config_path),
        "k_source": rt["k_source"],
        "model": {
            "nominal_gammas": g_nom,
            "uncertainty_pct": uncertainty_pct,
            "num_vertices": len(vertices),
        },
        "controller": {
            "control_sign": control_sign,
            "v_limits": v_limits.tolist(),
            "integral_limits": integral_limits.tolist(),
            "k_runtime_row_major": k_runtime.reshape(-1).tolist(),
            "k_index_map": k_index_map,
            "state_order": STATE_NAMES,
            "input_order": INPUT_NAMES,
        },
        "stability": stability,
        "metrics": metrics_all,
    }

    validation_yaml = out_dir / "lqi_validation.yaml"
    validation_yaml.write_text(yaml.safe_dump(result, sort_keys=False))

    summary_json = out_dir / "validation_summary.json"
    summary_json.write_text(json.dumps(result, indent=2))

    summary_txt = out_dir / "validation_summary.txt"
    lines = [
        "Robust LQI validation summary",
        f"config: {config_path}",
        f"k_source: {rt['k_source']}",
        f"out_dir: {out_dir}",
        "",
        "stability:",
        f"  stable_all_continuous: {stability['stable_all_continuous']}",
        f"  unstable_vertices_continuous: {stability['unstable_vertices_continuous']}",
        f"  worst_max_real_eig: {stability['worst_max_real_eig']:+.6f}",
        f"  worst_max_abs_eig_i_plus_a_dt: {stability['worst_max_abs_eig_i_plus_a_dt']:.6f}",
        "",
        f"validation_yaml: {validation_yaml}",
        f"summary_json: {summary_json}",
    ]
    summary_txt.write_text("\n".join(lines) + "\n")

    print("Robust LQI validation finished.")
    print(f"validation_yaml: {validation_yaml}")
    print(f"summary: {summary_txt}")
    print(f"stable_all_continuous: {stability['stable_all_continuous']}")
    print(f"worst_max_real_eig: {stability['worst_max_real_eig']:+.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate robust LQI gain from config and/or tuned YAML.")
    parser.add_argument("--config", required=True, help="Path to lqi config YAML.")
    parser.add_argument(
        "--tuned-yaml",
        default=None,
        help="Optional path to lqi_tuned.yaml (overrides controller.source_tuned_yaml).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory override for validation artifacts.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    tuned_yaml_override = Path(args.tuned_yaml).resolve() if args.tuned_yaml else None
    if tuned_yaml_override is not None and not tuned_yaml_override.exists():
        raise FileNotFoundError(f"Tuned YAML not found: {tuned_yaml_override}")

    out_dir_override = Path(args.out_dir).resolve() if args.out_dir else None

    run(config_path, tuned_yaml_override, out_dir_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
