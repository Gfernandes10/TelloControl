#!/usr/bin/env python3
"""Simulate the simplified drone plant from a YAML config.

Model (continuous-time):
  q_ddot = Lambda(psi) * q_dot + Gamma(psi) * v

State:
  q     = [x, y, z, yaw]
  q_dot = [vx, vy, vz, wyaw]
Input:
  v     = [v_x, v_y, v_z, v_yaw]

The script supports open-loop command generation modes:
  - constant
  - excitation
  - sequential_excitation
  - csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import yaml


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2, "yaw": 3}


def _as_float(value, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


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


def _vector4(data, key: str, default: List[float]) -> List[float]:
    v = data.get(key, default)
    if len(v) != 4:
        raise ValueError(f"'{key}' must have exactly 4 elements.")
    return [float(x) for x in v]


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _excitation_value(t: float, exc_cfg: Dict) -> float:
    scale = _as_float(exc_cfg.get("scale"), 0.1 / 4.5)
    coeffs = exc_cfg.get("coeffs", [3.0, 1.0, 0.5])
    omega_pi = exc_cfg.get("omega_pi", [0.2, 0.6, 1.0])
    if len(coeffs) != len(omega_pi):
        raise ValueError("excitation.coeffs and excitation.omega_pi must have same length")
    s = 0.0
    for c, w in zip(coeffs, omega_pi):
        s += float(c) * math.sin(float(w) * math.pi * t)
    return scale * s


class CsvCommandSource:
    def __init__(self, csv_path: Path, time_col: str, cmd_cols: List[str], hold_last: bool) -> None:
        self.times: List[float] = []
        self.cmds: List[List[float]] = []
        self.hold_last = hold_last

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row[time_col])
                cmd = [float(row[c]) for c in cmd_cols]
                self.times.append(t)
                self.cmds.append(cmd)

        if not self.times:
            raise ValueError(f"CSV command source is empty: {csv_path}")

    def sample(self, t: float) -> List[float]:
        if t <= self.times[0]:
            return self.cmds[0][:]
        if t >= self.times[-1]:
            if self.hold_last:
                return self.cmds[-1][:]
            return [0.0, 0.0, 0.0, 0.0]

        lo = 0
        hi = len(self.times) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.times[mid] <= t:
                lo = mid
            else:
                hi = mid

        t0 = self.times[lo]
        t1 = self.times[hi]
        a = (t - t0) / (t1 - t0)
        out = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            out[i] = (1.0 - a) * self.cmds[lo][i] + a * self.cmds[hi][i]
        return out


def _build_command_provider(cfg: Dict, cfg_dir: Path):
    input_cfg = cfg.get("input", {})
    mode = str(input_cfg.get("mode", "constant")).strip().lower()

    if mode == "constant":
        constant = _vector4(input_cfg, "constant", [0.0, 0.0, 0.0, 0.0])

        def provider(t: float) -> List[float]:  # pylint: disable=unused-argument
            return constant[:]

        return provider

    if mode == "excitation":
        weights = _vector4(input_cfg, "weights", [1.0, 0.0, 0.0, 0.0])
        exc_cfg = input_cfg.get("excitation", {})

        def provider(t: float) -> List[float]:
            u = _excitation_value(t, exc_cfg)
            return [u * weights[0], u * weights[1], u * weights[2], u * weights[3]]

        return provider

    if mode == "sequential_excitation":
        axes = input_cfg.get("sequential_axes", ["x", "y", "z", "yaw"])
        axis_idx: List[int] = []
        for a in axes:
            name = str(a).strip().lower()
            if name not in AXIS_TO_INDEX:
                raise ValueError(f"Invalid axis '{a}' in sequential_axes")
            axis_idx.append(AXIS_TO_INDEX[name])

        seg_dur = _as_float(input_cfg.get("segment_duration"), 8.0)
        repeat = _as_bool(input_cfg.get("repeat"), False)
        phase_reset = _as_bool(input_cfg.get("phase_reset"), True)
        exc_cfg = input_cfg.get("excitation", {})

        if seg_dur <= 0.0:
            raise ValueError("input.segment_duration must be > 0")
        if not axis_idx:
            raise ValueError("input.sequential_axes must not be empty")

        def provider(t: float) -> List[float]:
            seg = int(math.floor(t / seg_dur))
            if not repeat and seg >= len(axis_idx):
                return [0.0, 0.0, 0.0, 0.0]
            seg_mod = seg % len(axis_idx)
            local_t = t - seg * seg_dur if phase_reset else t
            u = _excitation_value(local_t, exc_cfg)
            out = [0.0, 0.0, 0.0, 0.0]
            out[axis_idx[seg_mod]] = u
            return out

        return provider

    if mode == "csv":
        csv_path_raw = input_cfg.get("csv_path")
        if not csv_path_raw:
            raise ValueError("input.csv_path is required when input.mode=csv")
        csv_path = Path(csv_path_raw)
        if not csv_path.is_absolute():
            csv_path = (cfg_dir / csv_path).resolve()

        time_col = str(input_cfg.get("time_column", "t"))
        cmd_cols = input_cfg.get("command_columns", ["cmd_vx", "cmd_vy", "cmd_vz", "cmd_wyaw"])
        if len(cmd_cols) != 4:
            raise ValueError("input.command_columns must have 4 columns")
        hold_last = _as_bool(input_cfg.get("hold_last"), True)

        source = CsvCommandSource(csv_path, time_col, [str(c) for c in cmd_cols], hold_last)

        def provider(t: float) -> List[float]:
            return source.sample(t)

        return provider

    raise ValueError(f"Unsupported input.mode '{mode}'")


def _plant_step(q_dot: List[float], v_cmd: List[float], psi: float, gammas: Dict[str, float]) -> List[float]:
    c = math.cos(psi)
    s = math.sin(psi)

    g1 = gammas["gamma1"]
    g2 = gammas["gamma2"]
    g3 = gammas["gamma3"]
    g4 = gammas["gamma4"]
    g5 = gammas["gamma5"]
    g6 = gammas["gamma6"]
    g7 = gammas["gamma7"]
    g8 = gammas["gamma8"]

    vx, vy, vz, wyaw = q_dot
    ux, uy, uz, uyaw = v_cmd

    qdd_x = (g2 * c) * vx + (-g4 * s) * vy + (g1 * c) * ux + (-g3 * s) * uy
    qdd_y = (g2 * s) * vx + (g4 * c) * vy + (g1 * s) * ux + (g3 * c) * uy
    qdd_z = g6 * vz + g5 * uz
    qdd_yaw = g8 * wyaw + g7 * uyaw

    return [qdd_x, qdd_y, qdd_z, qdd_yaw]


def simulate(config_path: Path) -> Optional[Path]:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg_dir = config_path.parent

    model_cfg = cfg.get("model", {})
    gammas = {
        "gamma1": _as_float(model_cfg.get("gamma1"), 3.75),
        "gamma2": _as_float(model_cfg.get("gamma2"), 1.10),
        "gamma3": _as_float(model_cfg.get("gamma3"), 3.75),
        "gamma4": _as_float(model_cfg.get("gamma4"), 1.10),
        "gamma5": _as_float(model_cfg.get("gamma5"), 2.68),
        "gamma6": _as_float(model_cfg.get("gamma6"), 0.75),
        "gamma7": _as_float(model_cfg.get("gamma7"), 1.42),
        "gamma8": _as_float(model_cfg.get("gamma8"), 2.06),
    }

    sim_cfg = cfg.get("simulation", {})
    dt = _as_float(sim_cfg.get("dt"), 0.2)
    duration = _as_float(sim_cfg.get("duration"), 40.0)
    psi_source = str(sim_cfg.get("psi_source", "state")).strip().lower()
    psi_fixed = _as_float(sim_cfg.get("psi_fixed"), 0.0)
    wrap_yaw = _as_bool(sim_cfg.get("wrap_yaw"), True)

    if dt <= 0.0:
        raise ValueError("simulation.dt must be > 0")
    if duration <= 0.0:
        raise ValueError("simulation.duration must be > 0")
    if psi_source not in ("state", "fixed"):
        raise ValueError("simulation.psi_source must be 'state' or 'fixed'")

    init_cfg = cfg.get("initial_state", {})
    q = [
        _as_float(init_cfg.get("x"), 0.0),
        _as_float(init_cfg.get("y"), 0.0),
        _as_float(init_cfg.get("z"), 0.0),
        _as_float(init_cfg.get("yaw"), 0.0),
    ]
    q_dot = [
        _as_float(init_cfg.get("vx"), 0.0),
        _as_float(init_cfg.get("vy"), 0.0),
        _as_float(init_cfg.get("vz"), 0.0),
        _as_float(init_cfg.get("wyaw"), 0.0),
    ]

    out_cfg = cfg.get("output", {})
    output_enabled = _as_bool(out_cfg.get("enabled"), True)
    output_csv_raw = out_cfg.get("csv_path", "plant_simulation.csv")
    output_csv = Path(output_csv_raw)
    if not output_csv.is_absolute():
        output_csv = (cfg_dir / output_csv).resolve()
    if output_enabled:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    command = _build_command_provider(cfg, cfg_dir)

    n_steps = int(math.floor(duration / dt))

    rows: List[Dict[str, float]] = []
    for k in range(n_steps + 1):
        t = k * dt
        v_cmd = command(t)
        psi = q[3] if psi_source == "state" else psi_fixed

        q_ddot = _plant_step(q_dot, v_cmd, psi, gammas)

        rows.append(
            {
                "t": t,
                "x": q[0],
                "y": q[1],
                "z": q[2],
                "yaw": q[3],
                "vx": q_dot[0],
                "vy": q_dot[1],
                "vz": q_dot[2],
                "wyaw": q_dot[3],
                "cmd_vx": v_cmd[0],
                "cmd_vy": v_cmd[1],
                "cmd_vz": v_cmd[2],
                "cmd_wyaw": v_cmd[3],
                "ax": q_ddot[0],
                "ay": q_ddot[1],
                "az": q_ddot[2],
                "awyaw": q_ddot[3],
                "psi_used": psi,
            }
        )

        # Explicit Euler integration.
        for i in range(4):
            q_dot[i] += q_ddot[i] * dt
        for i in range(4):
            q[i] += q_dot[i] * dt

        if wrap_yaw:
            q[3] = _wrap_pi(q[3])

    if output_enabled:
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "t",
                    "x",
                    "y",
                    "z",
                    "yaw",
                    "vx",
                    "vy",
                    "vz",
                    "wyaw",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_vz",
                    "cmd_wyaw",
                    "ax",
                    "ay",
                    "az",
                    "awyaw",
                    "psi_used",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("Plant simulation finished.")
    print(f"steps={n_steps + 1} dt={dt:.6f}s duration={duration:.3f}s")
    print(
        "final_state: "
        f"x={q[0]:.4f} y={q[1]:.4f} z={q[2]:.4f} yaw={q[3]:.4f} "
        f"vx={q_dot[0]:.4f} vy={q_dot[1]:.4f} vz={q_dot[2]:.4f} wyaw={q_dot[3]:.4f}"
    )
    if output_enabled:
        print(f"csv: {output_csv}")
        return output_csv

    print("csv: disabled (output.enabled=false)")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate simplified drone plant from YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config with model/simulation/input/output sections.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    simulate(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
