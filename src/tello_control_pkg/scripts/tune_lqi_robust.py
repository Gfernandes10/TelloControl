#!/usr/bin/env python3
"""Robust LQI tuning for the simplified drone model.

Pipeline:
1) Read one YAML config with model + tuning + validation settings.
2) Build robust polytopic vertices from gamma uncertainties.
3) Solve robust LQI-LMI (optionally with alpha and damping-ratio pole-region constraints) and compute K.
4) Optionally validate closed-loop stability/time-domain performance.
5) Save tuned gains and results.

Validation is enabled by default to preserve legacy behavior.
Use validation.run_after_tuning=false to skip validation and only synthesize.
For standalone validation (without re-synthesis), use validate_lqi_robust.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cvxpy as cp
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATE_NAMES = [
    "ex",
    "evx",
    "ey",
    "evy",
    "ez",
    "evz",
    "eyaw",
    "evyaw",
    "iex",
    "iey",
    "iez",
    "ieyaw",
]
INPUT_NAMES = ["ux", "uy", "uz", "uyaw"]
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2, "yaw": 3}


@dataclass
class Scenario:
    name: str
    axis: str
    amplitude: float
    step_time: float


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _as_float(value, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _log(message: str) -> None:
    print(f"[tune_lqi_robust] {message}", flush=True)


def _resolve_path(path_raw: str, cfg_dir: Path) -> Path:
    p = Path(path_raw)
    if not p.is_absolute():
        p = (cfg_dir / p).resolve()
    return p


def _build_ros_runtime_yaml(
    cfg: Dict,
    cfg_dir: Path,
    out_dir: Path,
    g_nom: Dict[str, float],
    control_sign: float,
    integral_limits: np.ndarray,
    v_limits: np.ndarray,
    k_runtime: np.ndarray,
) -> Path:
    out_cfg = cfg.get("output", {})
    node_name = str(out_cfg.get("ros_runtime_node_name", "/tello_control_node"))
    ros_runtime_yaml_raw = out_cfg.get("ros_runtime_yaml")
    ros_runtime_base_yaml_raw = out_cfg.get("ros_runtime_base_yaml")

    if ros_runtime_yaml_raw:
        ros_runtime_yaml = _resolve_path(str(ros_runtime_yaml_raw), cfg_dir)
    else:
        ros_runtime_yaml = out_dir / "lqi_runtime_ros.yaml"

    if ros_runtime_base_yaml_raw:
        base_path = _resolve_path(str(ros_runtime_base_yaml_raw), cfg_dir)
        if base_path.exists():
            base_data = yaml.safe_load(base_path.read_text()) or {}
            _log(f"Using ROS runtime base YAML: {base_path}")
        else:
            base_data = {}
            _log(
                f"ROS runtime base YAML not found ({base_path}); "
                "creating runtime YAML from scratch."
            )
    else:
        base_data = {}

    node_block = base_data.get(node_name)
    if not isinstance(node_block, dict):
        node_block = {}
        base_data[node_name] = node_block
    ros_params = node_block.get("ros__parameters")
    if not isinstance(ros_params, dict):
        ros_params = {}
        node_block["ros__parameters"] = ros_params

    # Map synthesis outputs to runtime ROS params.
    ros_params["controller_type"] = "lqi"
    ros_params["lqi.mode"] = "manual"
    ros_params["lqi.control_sign"] = float(control_sign)
    for i in range(1, 9):
        ros_params[f"lqi.gamma{i}"] = float(g_nom[f"gamma{i}"])

    int_xy = float(max(integral_limits[0], integral_limits[1]))
    if abs(float(integral_limits[0]) - float(integral_limits[1])) > 1e-12:
        _log(
            "Warning: integral_limits for X and Y differ; "
            f"using max for lqi.int_xy_limit ({int_xy:.6f})"
        )
    ros_params["lqi.int_xy_limit"] = int_xy
    ros_params["lqi.int_z_limit"] = float(integral_limits[2])
    ros_params["lqi.int_yaw_limit"] = float(integral_limits[3])

    cmd_xy = float(max(v_limits[0], v_limits[1]))
    if abs(float(v_limits[0]) - float(v_limits[1])) > 1e-12:
        _log(
            "Warning: v_limits for X and Y differ; "
            f"using max for lqi.cmd_xy_limit ({cmd_xy:.6f})"
        )
    ros_params["lqi.cmd_xy_limit"] = cmd_xy
    ros_params["lqi.cmd_z_limit"] = float(v_limits[2])
    ros_params["lqi.cmd_yaw_limit"] = float(v_limits[3])
    ros_params["lqi.k"] = k_runtime.reshape(-1).astype(float).tolist()

    ros_runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    ros_runtime_yaml.write_text(yaml.safe_dump(base_data, sort_keys=False))
    return ros_runtime_yaml


def _diag_from_cfg(data: Dict, key: str, expected_len: int, default: List[float]) -> np.ndarray:
    arr = data.get(key, default)
    if len(arr) != expected_len:
        raise ValueError(f"'{key}' must have length {expected_len}")
    vals = np.asarray([float(v) for v in arr], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError(f"'{key}' must contain strictly positive values")
    return np.diag(vals)


def _load_model_gammas(cfg: Dict, cfg_dir: Path) -> Dict[str, float]:
    model_cfg = cfg.get("model", {})
    gamma_keys = [f"gamma{i}" for i in range(1, 9)]
    source_yaml = model_cfg.get("source_yaml")

    if source_yaml:
        src = Path(source_yaml)
        if not src.is_absolute():
            src = (cfg_dir / src).resolve()
        data = yaml.safe_load(src.read_text()) or {}
        if "gammas" in data:
            source_data = data["gammas"]
        else:
            source_data = data
        loaded = {k: float(source_data[k]) for k in gamma_keys}
    else:
        loaded = {}

    direct = model_cfg.get("gammas", {})
    for k in gamma_keys:
        if k in direct:
            loaded[k] = float(direct[k])

    missing = [k for k in gamma_keys if k not in loaded]
    if missing:
        raise ValueError(f"Missing model gammas: {missing}")
    return loaded


def _build_vertices(nom: Dict[str, float], uncertainty_pct: float) -> List[Dict[str, float]]:
    if uncertainty_pct < 0.0:
        raise ValueError("model.uncertainty_pct must be >= 0")
    if uncertainty_pct == 0.0:
        return [{k: float(v) for k, v in nom.items()}]
    keys = [f"gamma{i}" for i in range(1, 9)]
    vertices: List[Dict[str, float]] = []
    for signs in itertools.product([-1.0, 1.0], repeat=8):
        g = {}
        for i, k in enumerate(keys):
            g[k] = nom[k] * (1.0 + signs[i] * uncertainty_pct)
        vertices.append(g)
    return vertices


def _build_augmented_matrices(g: Dict[str, float], g_nom: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    # State order:
    # [ex, evx, ey, evy, ez, evz, eyaw, evyaw, iex, iey, iez, ieyaw]
    # Input is virtual acceleration u = [ux, uy, uz, uyaw]
    a = np.zeros((12, 12), dtype=float)
    b = np.zeros((12, 4), dtype=float)

    a[0, 1] = 1.0
    a[1, 1] = g["gamma2"]
    a[2, 3] = 1.0
    a[3, 3] = g["gamma4"]
    a[4, 5] = 1.0
    a[5, 5] = g["gamma6"]
    a[6, 7] = 1.0
    a[7, 7] = g["gamma8"]

    a[8, 0] = 1.0
    a[9, 2] = 1.0
    a[10, 4] = 1.0
    a[11, 6] = 1.0

    # Effective virtual-input gain under model mismatch:
    # q_ddot = Lambda_actual q_dot + Gamma_actual Gamma_nom^-1 u
    b[1, 0] = g["gamma1"] / g_nom["gamma1"]
    b[3, 1] = g["gamma3"] / g_nom["gamma3"]
    b[5, 2] = g["gamma5"] / g_nom["gamma5"]
    b[7, 3] = g["gamma7"] / g_nom["gamma7"]
    return a, b


def _sqrt_psd_matrix(m: np.ndarray, eps: float) -> np.ndarray:
    ms = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(ms)
    vals = np.maximum(vals, eps)
    root = np.diag(np.sqrt(vals)) @ vecs.T
    return root


def _solve_robust_lqi(
    a_vertices: List[np.ndarray],
    b_vertices: List[np.ndarray],
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    solver: str,
    solver_eps: float,
    solver_max_iters: int,
    lmi_eps: float,
    alpha_margin: float,
    enable_damping_ratio: bool,
    zeta_min: float,
    enable_input_saturation_lmi: bool,
    u_limits_lmi: np.ndarray,
    objective_w2_weight: float,
    verbose: bool,
    clarabel_options: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    n = q.shape[0]
    m = r.shape[0]
    if s.shape != (n, m):
        raise ValueError("S must be n x m")

    c = np.block([[q, s], [s.T, r]])
    c = 0.5 * (c + c.T)
    t = _sqrt_psd_matrix(c, eps=lmi_eps)
    p = t.shape[0]
    t1 = t[:, :n]
    t2 = t[:, n:]

    w1 = cp.Variable((n, n), symmetric=True)
    w2 = cp.Variable((m, n))
    constraints = [w1 >> (lmi_eps * np.eye(n))]

    if enable_damping_ratio and not (0.0 < zeta_min < 1.0):
        raise ValueError("zeta_min must be in (0, 1) when damping-ratio constraint is enabled")

    sector_theta = math.acos(zeta_min) if enable_damping_ratio else 0.0
    sin_th = math.sin(sector_theta)
    cos_th = math.cos(sector_theta)

    if enable_input_saturation_lmi:
        if u_limits_lmi.shape != (m,):
            raise ValueError(f"u_limits_lmi must have shape ({m},)")
        if np.any(u_limits_lmi <= 0.0):
            raise ValueError("All u_limits_lmi entries must be > 0")
        for i in range(m):
            # Guarantees |u_i| <= u_limits_lmi[i] for states inside Lyapunov ellipsoid.
            yi = w2[i : i + 1, :]  # (1, n)
            sat_mtx = cp.bmat(
                [
                    [w1, yi.T],
                    [yi, np.array([[u_limits_lmi[i] ** 2]], dtype=float)],
                ]
            )
            constraints.append(sat_mtx >> (lmi_eps * np.eye(n + 1)))

    for a, b in zip(a_vertices, b_vertices):
        z = a @ w1 + b @ w2
        # Alpha-region: Re(lambda) < -alpha
        f = z + z.T + 2.0 * alpha_margin * w1
        g = w1 @ t1.T + w2.T @ t2.T
        h = t1 @ w1 + t2 @ w2
        mtx = cp.bmat([[f, g], [h, -np.eye(p)]])
        constraints.append(mtx << (-lmi_eps * np.eye(n + p)))

        # Damping-ratio conic sector constraint.
        if enable_damping_ratio:
            skew = z - z.T
            sector_mtx = cp.bmat(
                [
                    [sin_th * f, cos_th * skew],
                    [-cos_th * skew, sin_th * f],
                ]
            )
            constraints.append(sector_mtx << (-lmi_eps * np.eye(2 * n)))

    objective = cp.Minimize(cp.trace(w1) + objective_w2_weight * cp.norm(w2, "fro"))
    prob = cp.Problem(objective, constraints)

    solver_name = solver.upper()
    solve_kwargs: Dict[str, Any] = {"verbose": bool(verbose)}
    if solver_name == "SCS":
        solve_kwargs.update({"eps": solver_eps, "max_iters": solver_max_iters})
    elif solver_name == "CLARABEL":
        # Map generic knobs to Clarabel-specific options, then allow explicit overrides.
        solve_kwargs.update(
            {
                "max_iter": int(max(1, solver_max_iters)),
                "tol_gap_abs": float(solver_eps),
                "tol_gap_rel": float(solver_eps),
                "tol_feas": float(solver_eps),
                "tol_infeas_abs": float(solver_eps),
                "tol_infeas_rel": float(solver_eps),
            }
        )
        for key, value in clarabel_options.items():
            solve_kwargs[str(key)] = value
    prob.solve(solver=solver_name, **solve_kwargs)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LMI solve failed with status='{prob.status}'")

    w1_val = np.asarray(w1.value, dtype=float)
    w2_val = np.asarray(w2.value, dtype=float)
    k_lmi = w2_val @ np.linalg.inv(w1_val)
    return k_lmi, w1_val, str(prob.status), float(prob.value)


def _lambda_matrix(g: Dict[str, float], psi: float) -> np.ndarray:
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array(
        [
            [g["gamma2"] * c, -g["gamma4"] * s, 0.0, 0.0],
            [g["gamma2"] * s, g["gamma4"] * c, 0.0, 0.0],
            [0.0, 0.0, g["gamma6"], 0.0],
            [0.0, 0.0, 0.0, g["gamma8"]],
        ],
        dtype=float,
    )


def _gamma_matrix(g: Dict[str, float], psi: float) -> np.ndarray:
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array(
        [
            [g["gamma1"] * c, -g["gamma3"] * s, 0.0, 0.0],
            [g["gamma1"] * s, g["gamma3"] * c, 0.0, 0.0],
            [0.0, 0.0, g["gamma5"], 0.0],
            [0.0, 0.0, 0.0, g["gamma7"]],
        ],
        dtype=float,
    )


def _plant_step(q_dot: np.ndarray, v_cmd: np.ndarray, psi: float, g: Dict[str, float]) -> np.ndarray:
    c = math.cos(psi)
    s = math.sin(psi)
    vx, vy, vz, wyaw = q_dot.tolist()
    ux, uy, uz, uyaw = v_cmd.tolist()
    qdd_x = (g["gamma2"] * c) * vx + (-g["gamma4"] * s) * vy + (g["gamma1"] * c) * ux + (-g["gamma3"] * s) * uy
    qdd_y = (g["gamma2"] * s) * vx + (g["gamma4"] * c) * vy + (g["gamma1"] * s) * ux + (g["gamma3"] * c) * uy
    qdd_z = g["gamma6"] * vz + g["gamma5"] * uz
    qdd_yaw = g["gamma8"] * wyaw + g["gamma7"] * uyaw
    return np.asarray([qdd_x, qdd_y, qdd_z, qdd_yaw], dtype=float)


def _build_reference(s: Scenario, t: float) -> np.ndarray:
    r = np.zeros(4, dtype=float)
    if t >= s.step_time:
        r[AXIS_TO_INDEX[s.axis]] = s.amplitude
    return r


def _compute_metrics(
    t: np.ndarray,
    y: np.ndarray,
    ref: np.ndarray,
    step_time: float,
    settling_band: float,
) -> Dict[str, float]:
    final_ref = float(ref[-1])
    idx_step = int(np.searchsorted(t, step_time, side="left"))
    if idx_step >= len(t):
        idx_step = max(0, len(t) - 1)

    # Use pre-step reference level as baseline to avoid zero-amplitude artifacts
    # when step_time exactly matches a sampled timestamp.
    idx_before = max(0, idx_step - 1)
    ref_before = float(ref[idx_before])
    amp = final_ref - ref_before
    amp_abs = abs(amp)
    if amp_abs < 1e-9:
        # No effective step for this channel.
        return {
            "final_ref": final_ref,
            "step_amplitude": amp,
            "overshoot_abs": 0.0,
            "overshoot_pct": 0.0,
            "settling_time_s": 0.0,
        }

    y_after = y[idx_step:]
    ref_after = ref[idx_step:]

    if amp >= 0.0:
        peak = float(np.max(y_after))
        overshoot_abs = max(0.0, peak - final_ref)
    else:
        peak = float(np.min(y_after))
        overshoot_abs = max(0.0, final_ref - peak)
    overshoot_pct = 100.0 * overshoot_abs / amp_abs

    err = y_after - ref_after
    band = settling_band * amp_abs
    settle = float("nan")
    for i in range(len(err)):
        if np.all(np.abs(err[i:]) <= band):
            settle = float(t[idx_step + i] - step_time)
            break

    return {
        "final_ref": final_ref,
        "step_amplitude": amp,
        "overshoot_abs": overshoot_abs,
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settle,
    }


def _scenario_to_ref_amp(axis: str, amplitude: float, unit: str) -> float:
    if axis != "yaw":
        return float(amplitude)
    if unit.lower() == "deg":
        return float(math.radians(amplitude))
    if unit.lower() == "rad":
        return float(amplitude)
    raise ValueError("Yaw scenario unit must be 'deg' or 'rad'")


def _simulate_closed_loop(
    scenario: Scenario,
    g_plant: Dict[str, float],
    g_nom: Dict[str, float],
    k_runtime: np.ndarray,
    control_sign: float,
    dt: float,
    duration: float,
    integral_limits: np.ndarray,
    v_limits: np.ndarray,
    settling_band: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    n_steps = int(math.floor(duration / dt))
    q = np.zeros(4, dtype=float)
    q_dot = np.zeros(4, dtype=float)
    i_err = np.zeros(4, dtype=float)

    rows: List[Dict[str, float]] = []
    for k in range(n_steps + 1):
        t = k * dt
        ref = _build_reference(scenario, t)
        ref_dot = np.zeros(4, dtype=float)
        ref_ddot = np.zeros(4, dtype=float)

        e = q - ref
        e[3] = _wrap_pi(e[3])
        e_dot = q_dot - ref_dot
        x_aug = np.asarray(
            [
                e[0],
                e_dot[0],
                e[1],
                e_dot[1],
                e[2],
                e_dot[2],
                e[3],
                e_dot[3],
                i_err[0],
                i_err[1],
                i_err[2],
                i_err[3],
            ],
            dtype=float,
        )

        u_virtual = control_sign * (k_runtime @ x_aug)

        psi = float(q[3])
        lam = _lambda_matrix(g_nom, psi)
        gam = _gamma_matrix(g_nom, psi)
        rhs = u_virtual + ref_ddot - lam @ ref_dot
        v_raw = np.linalg.solve(gam, rhs)
        v_cmd = np.clip(v_raw, -v_limits, v_limits)

        q_ddot = _plant_step(q_dot, v_cmd, psi=psi, g=g_plant)

        rows.append(
            {
                "t": float(t),
                "x": float(q[0]),
                "y": float(q[1]),
                "z": float(q[2]),
                "yaw": float(q[3]),
                "vx": float(q_dot[0]),
                "vy": float(q_dot[1]),
                "vz": float(q_dot[2]),
                "wyaw": float(q_dot[3]),
                "x_ref": float(ref[0]),
                "y_ref": float(ref[1]),
                "z_ref": float(ref[2]),
                "yaw_ref": float(ref[3]),
                "u_x": float(u_virtual[0]),
                "u_y": float(u_virtual[1]),
                "u_z": float(u_virtual[2]),
                "u_yaw": float(u_virtual[3]),
                "v_x_raw": float(v_raw[0]),
                "v_y_raw": float(v_raw[1]),
                "v_z_raw": float(v_raw[2]),
                "v_yaw_raw": float(v_raw[3]),
                "v_x": float(v_cmd[0]),
                "v_y": float(v_cmd[1]),
                "v_z": float(v_cmd[2]),
                "v_yaw": float(v_cmd[3]),
            }
        )

        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt
        q[3] = _wrap_pi(float(q[3]))

        i_err = i_err + e * dt
        i_err = np.clip(i_err, -integral_limits, integral_limits)

    t_arr = np.asarray([r["t"] for r in rows], dtype=float)
    axis_col = {"x": "x", "y": "y", "z": "z", "yaw": "yaw"}[scenario.axis]
    axis_ref_col = {"x": "x_ref", "y": "y_ref", "z": "z_ref", "yaw": "yaw_ref"}[scenario.axis]
    y_arr = np.asarray([r[axis_col] for r in rows], dtype=float)
    ref_arr = np.asarray([r[axis_ref_col] for r in rows], dtype=float)
    base = _compute_metrics(t_arr, y_arr, ref_arr, scenario.step_time, settling_band=settling_band)

    v_abs = np.asarray(
        [[abs(r["v_x"]), abs(r["v_y"]), abs(r["v_z"]), abs(r["v_yaw"])] for r in rows],
        dtype=float,
    )
    u_abs = np.asarray(
        [[abs(r["u_x"]), abs(r["u_y"]), abs(r["u_z"]), abs(r["u_yaw"])] for r in rows],
        dtype=float,
    )
    sat = np.asarray(
        [
            [
                abs(r["v_x_raw"] - r["v_x"]) > 1e-9,
                abs(r["v_y_raw"] - r["v_y"]) > 1e-9,
                abs(r["v_z_raw"] - r["v_z"]) > 1e-9,
                abs(r["v_yaw_raw"] - r["v_yaw"]) > 1e-9,
            ]
            for r in rows
        ],
        dtype=bool,
    )

    for i, name in enumerate(INPUT_NAMES):
        base[f"peak_abs_u_{name}"] = float(np.max(u_abs[:, i]))
        base[f"peak_abs_v_{name}"] = float(np.max(v_abs[:, i]))
        base[f"sat_ratio_{name}"] = float(np.mean(sat[:, i]))

    return base, rows


def _plot_case(rows: List[Dict[str, float]], title: str, png_path: Path) -> None:
    t = np.asarray([r["t"] for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
    pos_cols = [("x", "x_ref"), ("y", "y_ref"), ("z", "z_ref"), ("yaw", "yaw_ref")]
    vel_cols = ["vx", "vy", "vz", "wyaw"]
    v_cols = ["v_x", "v_y", "v_z", "v_yaw"]
    u_cols = ["u_x", "u_y", "u_z", "u_yaw"]

    for i, (state_col, ref_col) in enumerate(pos_cols):
        ax = axes[0, i]
        ax.plot(t, [r[state_col] for r in rows], label="state", linewidth=1.2)
        ax.plot(t, [r[ref_col] for r in rows], "--", label="ref", linewidth=1.2)
        ax.set_title(state_col)
        ax.grid(True, alpha=0.3)

    for i, col in enumerate(vel_cols):
        ax = axes[1, i]
        ax.plot(t, [r[col] for r in rows], linewidth=1.2)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    for i, vcol in enumerate(v_cols):
        ax = axes[2, i]
        ax.plot(t, [r[vcol] for r in rows], color="tab:green", linewidth=1.2)
        ax.set_title(f"{vcol}_cmd")
        ax.grid(True, alpha=0.3)

    for i, ucol in enumerate(u_cols):
        ax = axes[3, i]
        ax.plot(t, [r[ucol] for r in rows], color="tab:purple", linewidth=1.2)
        ax.set_title(f"{ucol}_virtual")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def _write_rows_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _evaluate_linear_stability(
    a_vertices: List[np.ndarray],
    b_vertices: List[np.ndarray],
    k_runtime: np.ndarray,
    control_sign: float,
    dt: float,
) -> Dict[str, float]:
    max_real = -1e9
    max_abs_disc = -1e9
    unstable_cont = 0
    for a, b in zip(a_vertices, b_vertices):
        acl = a + b @ (control_sign * k_runtime)
        eig = np.linalg.eigvals(acl)
        real_max = float(np.max(np.real(eig)))
        max_real = max(max_real, real_max)
        if real_max >= 0.0:
            unstable_cont += 1

        ed = np.linalg.eigvals(np.eye(acl.shape[0]) + acl * dt)
        max_abs_disc = max(max_abs_disc, float(np.max(np.abs(ed))))

    return {
        "worst_max_real_eig": max_real,
        "worst_max_abs_eig_i_plus_a_dt": max_abs_disc,
        "unstable_vertices_continuous": unstable_cont,
        "stable_all_continuous": bool(unstable_cont == 0),
    }


def run(config_path: Path) -> None:
    t0 = time.perf_counter()
    _log(f"Loading config: {config_path}")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg_dir = config_path.parent

    g_nom = _load_model_gammas(cfg, cfg_dir)
    _log(
        "Loaded nominal gammas: "
        + ", ".join([f"{k}={v:+.6f}" for k, v in g_nom.items()])
    )

    model_cfg = cfg.get("model", {})
    uncertainty_pct = _as_float(model_cfg.get("uncertainty_pct"), 0.20)
    vertices = _build_vertices(g_nom, uncertainty_pct)
    _log(f"Generated {len(vertices)} uncertainty vertices (uncertainty_pct={uncertainty_pct:.3f})")
    a_vertices = []
    b_vertices = []
    progress_step = max(1, len(vertices) // 10)
    for i, gv in enumerate(vertices):
        a, b = _build_augmented_matrices(gv, g_nom)
        a_vertices.append(a)
        b_vertices.append(b)
        if (i + 1) % progress_step == 0 or (i + 1) == len(vertices):
            _log(f"Built augmented matrices for {i + 1}/{len(vertices)} vertices")

    ctrl_cfg = cfg.get("controller", {})
    control_sign = _as_float(ctrl_cfg.get("control_sign"), -1.0)
    integral_limits = np.asarray(ctrl_cfg.get("integral_limits", [0.8, 0.8, 0.8, 1.2]), dtype=float)
    if integral_limits.shape != (4,):
        raise ValueError("controller.integral_limits must have 4 elements")
    v_limits = np.asarray(ctrl_cfg.get("v_limits", [0.5, 0.5, 0.5, 0.5]), dtype=float)
    if v_limits.shape != (4,):
        raise ValueError("controller.v_limits must have 4 elements")

    q = _diag_from_cfg(ctrl_cfg, "q_diag", 12, [12.0, 3.0, 12.0, 3.0, 16.0, 4.0, 10.0, 2.0, 2.0, 2.0, 3.0, 2.0])
    r = _diag_from_cfg(ctrl_cfg, "r_diag", 4, [1.0, 1.0, 1.0, 1.0])
    s = np.zeros((12, 4), dtype=float)

    synth_cfg = cfg.get("synthesis", {})
    solver = str(synth_cfg.get("solver", "CLARABEL"))
    solver_eps = _as_float(synth_cfg.get("solver_eps"), 1e-5)
    solver_max_iters = _as_int(synth_cfg.get("solver_max_iters"), 25000)
    verbose = _as_bool(synth_cfg.get("verbose"), False)
    clarabel_options_raw = synth_cfg.get("clarabel_options", {})
    if clarabel_options_raw is None:
        clarabel_options_raw = {}
    if not isinstance(clarabel_options_raw, dict):
        raise ValueError("synthesis.clarabel_options must be a mapping/dict")
    clarabel_options: Dict[str, Any] = dict(clarabel_options_raw)
    lmi_eps = _as_float(synth_cfg.get("lmi_eps"), 1e-7)
    pole_region_cfg = synth_cfg.get("pole_region", {})
    enable_alpha = _as_bool(pole_region_cfg.get("enable_alpha"), False)
    alpha_margin = _as_float(pole_region_cfg.get("alpha"), 0.0) if enable_alpha else 0.0
    enable_damping_ratio = _as_bool(pole_region_cfg.get("enable_damping_ratio"), False)
    zeta_min = _as_float(pole_region_cfg.get("zeta_min"), 0.7) if enable_damping_ratio else 0.0
    sat_lmi_cfg = synth_cfg.get("input_saturation_lmi", {})
    enable_input_saturation_lmi = _as_bool(sat_lmi_cfg.get("enable"), False)
    u_limits_lmi = np.asarray(sat_lmi_cfg.get("u_limits", [1.0, 1.0, 1.0, 1.0]), dtype=float)

    # Backward compatibility (old key).
    if "decay_margin" in synth_cfg and not enable_alpha:
        alpha_margin = _as_float(synth_cfg.get("decay_margin"), 0.0)

    objective_w2_weight = _as_float(synth_cfg.get("objective_w2_weight"), 1e-4)
    _log(
        "Synthesis settings: "
        f"solver={solver}, solver_eps={solver_eps}, solver_max_iters={solver_max_iters}, "
        f"lmi_eps={lmi_eps}, alpha_enabled={enable_alpha}, damping_enabled={enable_damping_ratio}, "
        f"input_saturation_lmi_enabled={enable_input_saturation_lmi}, verbose={verbose}"
    )
    if solver.upper() == "CLARABEL" and clarabel_options:
        _log(f"Clarabel options override: {clarabel_options}")
    _log("Starting robust LQI LMI solve...")
    t_solve = time.perf_counter()

    k_lmi, w1, solve_status, objective_value = _solve_robust_lqi(
        a_vertices=a_vertices,
        b_vertices=b_vertices,
        q=q,
        r=r,
        s=s,
        solver=solver,
        solver_eps=solver_eps,
        solver_max_iters=solver_max_iters,
        lmi_eps=lmi_eps,
        alpha_margin=alpha_margin,
        enable_damping_ratio=enable_damping_ratio,
        zeta_min=zeta_min,
        enable_input_saturation_lmi=enable_input_saturation_lmi,
        u_limits_lmi=u_limits_lmi,
        objective_w2_weight=objective_w2_weight,
        verbose=verbose,
        clarabel_options=clarabel_options,
    )
    solve_elapsed = time.perf_counter() - t_solve
    _log(
        f"LMI solve finished: status={solve_status}, objective={objective_value:.6e}, "
        f"elapsed={solve_elapsed:.2f}s"
    )

    # Runtime convention:
    # u = control_sign * K_runtime * x
    # We synthesize K_lmi for u = K_lmi * x -> K_runtime = K_lmi / control_sign
    if abs(control_sign) < 1e-12:
        raise ValueError("controller.control_sign must be non-zero")
    k_runtime = k_lmi / control_sign

    out_cfg = cfg.get("output", {})
    out_dir_raw = out_cfg.get("dir", "src/tello_control_pkg/experiments_lqi/latest")
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        out_dir = (cfg_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_plots = _as_bool(out_cfg.get("save_plots"), True)
    save_csv = _as_bool(out_cfg.get("save_csv"), True)

    val_cfg = cfg.get("validation", {})
    run_after_tuning = _as_bool(val_cfg.get("run_after_tuning"), True)
    simulate_worst_vertex = _as_bool(val_cfg.get("simulate_worst_vertex"), True)

    stability: Dict[str, float] = {}
    metrics_all: Dict[str, Dict[str, float]] = {}
    if run_after_tuning:
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
        _log(
            f"Validation enabled: dt={dt:.4f}s, duration={duration:.2f}s, "
            f"settling_band={settling_band:.4f}, scenarios={len(scenarios)}"
        )

        _log("Evaluating closed-loop linear stability over vertices...")
        stability = _evaluate_linear_stability(a_vertices, b_vertices, k_runtime, control_sign, dt)
        _log(
            "Stability summary: "
            f"stable_all_continuous={stability['stable_all_continuous']}, "
            f"worst_max_real_eig={stability['worst_max_real_eig']:+.6f}"
        )

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
        _log(f"Validation cases: {[c[0] for c in cases]}")

        for case_name, g_case in cases:
            for sc in scenarios:
                _log(f"Simulating case='{case_name}' scenario='{sc.name}'...")
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
                _log(
                    f"Completed {key}: overshoot={m['overshoot_pct']:.3f}% "
                    f"settling={m['settling_time_s']:.3f}s "
                    f"peak_vz={m.get('peak_abs_v_uz', float('nan')):.4f}"
                )

                if save_csv:
                    _write_rows_csv(rows, out_dir / "csv" / f"{key}.csv")
                if save_plots:
                    _plot_case(rows, title=key, png_path=out_dir / "plots" / f"{key}.png")
        _log("Validation phase completed.")
    else:
        _log("Validation phase skipped (validation.run_after_tuning=false).")

    k_index_map = []
    for r_idx, u_name in enumerate(INPUT_NAMES):
        for c_idx, st_name in enumerate(STATE_NAMES):
            flat_idx = r_idx * 12 + c_idx
            k_index_map.append(f"k[{flat_idx:02d}] = K[{r_idx},{c_idx}] -> {u_name} <- {st_name}")

    result = {
        "config": str(config_path),
        "solver": {"status": solve_status, "objective": objective_value},
        "synthesis": {
            "solver": solver,
            "solver_eps": solver_eps,
            "solver_max_iters": solver_max_iters,
            "verbose": bool(verbose),
            "clarabel_options": clarabel_options if solver.upper() == "CLARABEL" else {},
            "lmi_eps": lmi_eps,
            "pole_region": {
                "enable_alpha": bool(enable_alpha),
                "alpha": float(alpha_margin),
                "enable_damping_ratio": bool(enable_damping_ratio),
                "zeta_min": float(zeta_min) if enable_damping_ratio else None,
            },
            "input_saturation_lmi": {
                "enable": bool(enable_input_saturation_lmi),
                "u_limits": u_limits_lmi.tolist() if enable_input_saturation_lmi else None,
            },
            "objective_w2_weight": objective_w2_weight,
        },
        "model": {
            "nominal_gammas": g_nom,
            "uncertainty_pct": uncertainty_pct,
            "num_vertices": len(vertices),
        },
        "controller": {
            "control_sign": control_sign,
            "v_limits": v_limits.tolist(),
            "integral_limits": integral_limits.tolist(),
            "k_lmi_row_major": k_lmi.reshape(-1).tolist(),
            "k_runtime_row_major": k_runtime.reshape(-1).tolist(),
            "k_index_map": k_index_map,
            "state_order": STATE_NAMES,
            "input_order": INPUT_NAMES,
        },
        "notes": {
            "validation_run_after_tuning": bool(run_after_tuning),
            "validation_script": "validate_lqi_robust.py",
        },
    }
    if run_after_tuning:
        result["stability"] = stability
        result["metrics"] = metrics_all

    ros_runtime_yaml = _build_ros_runtime_yaml(
        cfg=cfg,
        cfg_dir=cfg_dir,
        out_dir=out_dir,
        g_nom=g_nom,
        control_sign=control_sign,
        integral_limits=integral_limits,
        v_limits=v_limits,
        k_runtime=k_runtime,
    )
    result["ros_runtime_yaml"] = str(ros_runtime_yaml)

    tuned_yaml = out_dir / "lqi_tuned.yaml"
    _log(f"Writing artifacts to: {out_dir}")
    tuned_yaml.write_text(yaml.safe_dump(result, sort_keys=False))

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(result, indent=2))

    summary_txt = out_dir / "summary.txt"
    lines = [
        "Robust LQI tuning summary",
        f"config: {config_path}",
        f"out_dir: {out_dir}",
        "",
        f"solver_status: {solve_status}",
        f"solver_objective: {objective_value:.6e}",
        f"solver: {solver}",
        f"verbose: {verbose}",
        f"alpha_enabled: {enable_alpha}",
        f"alpha: {alpha_margin:.6f}",
        f"damping_ratio_enabled: {enable_damping_ratio}",
        f"zeta_min: {zeta_min if enable_damping_ratio else 'disabled'}",
        f"input_saturation_lmi_enabled: {enable_input_saturation_lmi}",
        f"u_limits_lmi: {u_limits_lmi.tolist() if enable_input_saturation_lmi else 'disabled'}",
        f"vertices: {len(vertices)}",
        "",
        f"validation_run_after_tuning: {run_after_tuning}",
        "",
        f"tuned_yaml: {tuned_yaml}",
        f"ros_runtime_yaml: {ros_runtime_yaml}",
        f"summary_json: {summary_json}",
    ]
    if run_after_tuning:
        lines.extend(
            [
                "",
                "stability:",
                f"  stable_all_continuous: {stability['stable_all_continuous']}",
                f"  unstable_vertices_continuous: {stability['unstable_vertices_continuous']}",
                f"  worst_max_real_eig: {stability['worst_max_real_eig']:+.6f}",
                f"  worst_max_abs_eig_i_plus_a_dt: {stability['worst_max_abs_eig_i_plus_a_dt']:.6f}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "closed_loop_validation: skipped in this run",
                "run validate_lqi_robust.py for stability and time-domain metrics",
            ]
        )
    summary_txt.write_text("\n".join(lines) + "\n")

    print("Robust LQI tuning finished.")
    print(f"tuned_yaml: {tuned_yaml}")
    print(f"ros_runtime_yaml: {ros_runtime_yaml}")
    print(f"summary: {summary_txt}")
    if run_after_tuning:
        print(f"stable_all_continuous: {stability['stable_all_continuous']}")
        print(f"worst_max_real_eig: {stability['worst_max_real_eig']:+.6f}")
    else:
        print("Closed-loop validation skipped in this run.")
        print("Use validate_lqi_robust.py to evaluate stability and step responses.")
    _log(f"Total elapsed time: {time.perf_counter() - t0:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune robust LQI from one YAML config.")
    parser.add_argument("--config", required=True, help="Path to lqi tuning YAML.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    run(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
