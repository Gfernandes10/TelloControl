#!/usr/bin/env python3
"""Identify simplified drone plant parameters from CSV experiments.

Expected structure under --root (both are supported):
  1) <any_experiment_dir>/tello_interface/{filtered_pose.csv,u_control.csv}
  2) Flat merged CSV per experiment containing columns like:
     filtered_pose/* and u_control/* (e.g. ExpTodos_manual_x.csv)

Axis for each experiment:
  1) If folder name matches ExpX/ExpY/ExpZ/ExpYaw, axis comes from the name.
  2) Otherwise, axis is inferred from command RMS among [ux, uy, uz, uyaw].

Model:
  q_ddot = Lambda(psi) q_dot + Gamma(psi) v

with parameters gamma1..gamma8.

Outputs (inside --root by default):
  - identified_gammas.yaml
  - identification_summary.txt
  - plots/<experiment>_model_vs_measured.png
  - plots/<experiment>_model_vs_measured.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "yaw": 3}
EXP_RE = re.compile(r"^Exp(X|Y|Z|Yaw)_", re.IGNORECASE)


@dataclass
class ExperimentData:
    name: str
    axis: str
    axis_source: str
    base_dir: Path
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    wyaw: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    uyaw: np.ndarray


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    out = angle.copy()
    out = (out + np.pi) % (2.0 * np.pi) - np.pi
    return out


def _safe_gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.zeros_like(y)
    dy = np.gradient(y, t)
    dy[~np.isfinite(dy)] = 0.0
    return dy


def _world_to_body_xy(vx_w: np.ndarray, vy_w: np.ndarray, yaw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = np.cos(yaw)
    s = np.sin(yaw)
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b


def _zoh_sample(times: np.ndarray, values: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Zero-order-hold resampling.

    times: (N,), strictly increasing
    values: (N, D)
    query_t: (M,)
    return: (M, D)
    """
    out = np.zeros((len(query_t), values.shape[1]), dtype=float)
    if len(times) == 0:
        return out

    j = 0
    for i, tq in enumerate(query_t):
        while j + 1 < len(times) and times[j + 1] <= tq:
            j += 1
        out[i, :] = values[j, :]
    return out


def _read_filtered_pose(csv_path: Path) -> Dict[str, np.ndarray]:
    t, x, y, z, yaw, vx, vy, vz, wyaw = ([] for _ in range(9))

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tt = float(row.get("timestamp", "nan"))
            except ValueError:
                continue
            if not math.isfinite(tt):
                continue

            def g(name: str, default: float = 0.0) -> float:
                try:
                    return float(row.get(name, default))
                except (ValueError, TypeError):
                    return float(default)

            t.append(tt)
            x.append(g("x"))
            y.append(g("y"))
            z.append(g("z"))
            yaw.append(g("yaw"))
            vx.append(g("dx"))
            vy.append(g("dy"))
            vz.append(g("dz"))

            if row.get("r") not in (None, ""):
                wyaw.append(g("r"))
            else:
                wyaw.append(float("nan"))

    t_arr = np.asarray(t, dtype=float)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]

    out = {
        "t": t_arr,
        "x": np.asarray(x, dtype=float)[order],
        "y": np.asarray(y, dtype=float)[order],
        "z": np.asarray(z, dtype=float)[order],
        "yaw": np.asarray(yaw, dtype=float)[order],
        "vx": np.asarray(vx, dtype=float)[order],
        "vy": np.asarray(vy, dtype=float)[order],
        "vz": np.asarray(vz, dtype=float)[order],
        "wyaw": np.asarray(wyaw, dtype=float)[order],
    }

    # If r was missing/invalid, derive wyaw from yaw.
    if np.isnan(out["wyaw"]).all():
        out["wyaw"] = _safe_gradient(_wrap_pi(out["yaw"]), out["t"])
    else:
        bad = ~np.isfinite(out["wyaw"])
        if np.any(bad):
            out["wyaw"][bad] = 0.0

    return out


def _read_u_control(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    t_list: List[float] = []
    u_list: List[List[float]] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tt = float(row.get("timestamp", "nan"))
            except ValueError:
                continue
            if not math.isfinite(tt):
                continue

            def g(name: str) -> float:
                try:
                    return float(row.get(name, 0.0))
                except (ValueError, TypeError):
                    return 0.0

            t_list.append(tt)
            u_list.append([g("ux"), g("uy"), g("uz"), g("uyaw")])

    if not t_list:
        return np.zeros((0,), dtype=float), np.zeros((0, 4), dtype=float)

    t = np.asarray(t_list, dtype=float)
    u = np.asarray(u_list, dtype=float)
    order = np.argsort(t)
    return t[order], u[order, :]


def _axis_from_name(name: str) -> Optional[str]:
    m = EXP_RE.match(name)
    if m:
        axis_raw = m.group(1).lower()
        return "yaw" if axis_raw == "yaw" else axis_raw

    low = name.lower()
    if "_yaw" in low or low.endswith("yaw"):
        return "yaw"
    if "_x" in low or low.endswith("x"):
        return "x"
    if "_y" in low or low.endswith("y"):
        return "y"
    if "_z" in low or low.endswith("z"):
        return "z"
    return None


def _read_combined_experiment_csv(csv_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    pose_t: List[float] = []
    x: List[float] = []
    y: List[float] = []
    z: List[float] = []
    yaw: List[float] = []
    vx: List[float] = []
    vy: List[float] = []
    vz: List[float] = []
    wyaw: List[float] = []

    u_t: List[float] = []
    u_vals: List[List[float]] = []

    with csv_path.open("r", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
          def g(name: str, default: float = float("nan")) -> float:
              try:
                  raw = row.get(name, default)
                  if raw in (None, ""):
                      return float(default)
                  return float(raw)
              except (ValueError, TypeError):
                  return float(default)

          # Pose sample timestamp priority:
          # filtered_pose/timestamp -> timestamp -> __time -> time
          t_pose = g("filtered_pose/timestamp", float("nan"))
          if not math.isfinite(t_pose):
              t_pose = g("timestamp", float("nan"))
          if not math.isfinite(t_pose):
              t_pose = g("__time", float("nan"))
          if not math.isfinite(t_pose):
              t_pose = g("time", float("nan"))

          if math.isfinite(t_pose):
              xx = g("filtered_pose/x", float("nan"))
              yy = g("filtered_pose/y", float("nan"))
              zz = g("filtered_pose/z", float("nan"))
              yw = g("filtered_pose/yaw", float("nan"))
              vxx = g("filtered_pose/dx", float("nan"))
              vyy = g("filtered_pose/dy", float("nan"))
              vzz = g("filtered_pose/dz", float("nan"))
              rr = g("filtered_pose/r", float("nan"))
              if all(math.isfinite(v) for v in [xx, yy, zz, yw, vxx, vyy, vzz]):
                  pose_t.append(t_pose)
                  x.append(xx)
                  y.append(yy)
                  z.append(zz)
                  yaw.append(yw)
                  vx.append(vxx)
                  vy.append(vyy)
                  vz.append(vzz)
                  wyaw.append(rr)

          # Command stream (sparse in merged CSV).
          t_u = g("u_control/timestamp", float("nan"))
          if math.isfinite(t_u):
              ux = g("u_control/ux", 0.0)
              uy = g("u_control/uy", 0.0)
              uz = g("u_control/uz", 0.0)
              uyaw = g("u_control/uyaw", 0.0)
              u_t.append(t_u)
              u_vals.append([ux, uy, uz, uyaw])

    if not pose_t:
        fp = {
            "t": np.zeros((0,), dtype=float),
            "x": np.zeros((0,), dtype=float),
            "y": np.zeros((0,), dtype=float),
            "z": np.zeros((0,), dtype=float),
            "yaw": np.zeros((0,), dtype=float),
            "vx": np.zeros((0,), dtype=float),
            "vy": np.zeros((0,), dtype=float),
            "vz": np.zeros((0,), dtype=float),
            "wyaw": np.zeros((0,), dtype=float),
        }
        return fp, np.zeros((0,), dtype=float), np.zeros((0, 4), dtype=float)

    t_arr = np.asarray(pose_t, dtype=float)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    fp = {
        "t": t_arr,
        "x": np.asarray(x, dtype=float)[order],
        "y": np.asarray(y, dtype=float)[order],
        "z": np.asarray(z, dtype=float)[order],
        "yaw": np.asarray(yaw, dtype=float)[order],
        "vx": np.asarray(vx, dtype=float)[order],
        "vy": np.asarray(vy, dtype=float)[order],
        "vz": np.asarray(vz, dtype=float)[order],
        "wyaw": np.asarray(wyaw, dtype=float)[order],
    }

    if np.isnan(fp["wyaw"]).all():
        fp["wyaw"] = _safe_gradient(_wrap_pi(fp["yaw"]), fp["t"])
    else:
        bad = ~np.isfinite(fp["wyaw"])
        if np.any(bad):
            fp["wyaw"][bad] = 0.0

    if not u_t:
        return fp, np.zeros((0,), dtype=float), np.zeros((0, 4), dtype=float)

    tu = np.asarray(u_t, dtype=float)
    uu = np.asarray(u_vals, dtype=float)
    u_order = np.argsort(tu)
    return fp, tu[u_order], uu[u_order, :]


def _trim_by_active_input(t: np.ndarray, u_axis: np.ndarray, threshold: float, pad_s: float) -> np.ndarray:
    idx = np.where(np.abs(u_axis) >= threshold)[0]
    if len(idx) == 0:
        return np.ones_like(t, dtype=bool)
    t0 = t[idx[0]] - pad_s
    t1 = t[idx[-1]] + pad_s
    return (t >= t0) & (t <= t1)


def _infer_axis_from_input(u: np.ndarray) -> Optional[str]:
    if u.size == 0 or u.shape[1] != 4:
        return None
    rms = np.sqrt(np.mean(np.square(u), axis=0))
    idx = int(np.argmax(rms))
    if not np.isfinite(rms[idx]) or rms[idx] < 1e-6:
        return None
    return ["x", "y", "z", "yaw"][idx]


def _load_experiment(exp_dir: Path, active_threshold: float, pad_s: float) -> Optional[ExperimentData]:
    axis_from_name: Optional[str] = _axis_from_name(exp_dir.name)

    ti_dir = exp_dir / "tello_interface"
    filtered_path = ti_dir / "filtered_pose.csv"
    u_path = ti_dir / "u_control.csv"
    if not filtered_path.exists() or not u_path.exists():
        return None

    fp = _read_filtered_pose(filtered_path)
    tu, uu = _read_u_control(u_path)
    if len(fp["t"]) < 5 or len(tu) == 0:
        return None

    t = fp["t"]
    u_res = _zoh_sample(tu, uu, t)

    axis = axis_from_name
    axis_source = "folder_name"
    if axis is None:
        axis = _infer_axis_from_input(u_res)
        axis_source = "input_rms"
    if axis is None:
        return None

    mask = _trim_by_active_input(t, u_res[:, AXIS_INDEX[axis]], active_threshold, pad_s)
    if np.count_nonzero(mask) < 5:
        mask = np.ones_like(t, dtype=bool)

    return ExperimentData(
        name=exp_dir.name,
        axis=axis,
        axis_source=axis_source,
        base_dir=exp_dir,
        t=t[mask],
        x=fp["x"][mask],
        y=fp["y"][mask],
        z=fp["z"][mask],
        yaw=fp["yaw"][mask],
        vx=fp["vx"][mask],
        vy=fp["vy"][mask],
        vz=fp["vz"][mask],
        wyaw=fp["wyaw"][mask],
        ux=u_res[mask, 0],
        uy=u_res[mask, 1],
        uz=u_res[mask, 2],
        uyaw=u_res[mask, 3],
    )


def _load_experiment_from_combined_csv(
    csv_path: Path,
    active_threshold: float,
    pad_s: float,
) -> Optional[ExperimentData]:
    axis_from_name: Optional[str] = _axis_from_name(csv_path.stem)

    fp, tu, uu = _read_combined_experiment_csv(csv_path)
    if len(fp["t"]) < 5 or len(tu) == 0:
        return None

    t = fp["t"]
    u_res = _zoh_sample(tu, uu, t)

    axis = axis_from_name
    axis_source = "file_name"
    if axis is None:
        axis = _infer_axis_from_input(u_res)
        axis_source = "input_rms"
    if axis is None:
        return None

    mask = _trim_by_active_input(t, u_res[:, AXIS_INDEX[axis]], active_threshold, pad_s)
    if np.count_nonzero(mask) < 5:
        mask = np.ones_like(t, dtype=bool)

    return ExperimentData(
        name=csv_path.stem,
        axis=axis,
        axis_source=axis_source,
        base_dir=csv_path.parent,
        t=t[mask],
        x=fp["x"][mask],
        y=fp["y"][mask],
        z=fp["z"][mask],
        yaw=fp["yaw"][mask],
        vx=fp["vx"][mask],
        vy=fp["vy"][mask],
        vz=fp["vz"][mask],
        wyaw=fp["wyaw"][mask],
        ux=u_res[mask, 0],
        uy=u_res[mask, 1],
        uz=u_res[mask, 2],
        uyaw=u_res[mask, 3],
    )


def load_all_experiments(root: Path, active_threshold: float, pad_s: float) -> List[ExperimentData]:
    experiments: List[ExperimentData] = []
    seen: set = set()

    # Format 1: split CSVs in tello_interface/ directory.
    for filtered_path in sorted(root.rglob("filtered_pose.csv")):
        ti_dir = filtered_path.parent
        if ti_dir.name != "tello_interface":
            continue
        exp_dir = ti_dir.parent
        if exp_dir in seen:
            continue
        seen.add(exp_dir)
        exp = _load_experiment(exp_dir, active_threshold, pad_s)
        if exp is not None:
            experiments.append(exp)

    # Format 2: merged single CSV per experiment.
    for csv_path in sorted(root.glob("*.csv")):
        key = ("csv", csv_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        exp = _load_experiment_from_combined_csv(csv_path, active_threshold, pad_s)
        if exp is not None:
            experiments.append(exp)

    return experiments


def fit_xy(experiments: List[ExperimentData]) -> Tuple[np.ndarray, Dict[str, float]]:
    a_blocks: List[np.ndarray] = []
    b_blocks: List[np.ndarray] = []

    for exp in experiments:
        if exp.axis not in ("x", "y"):
            continue

        t = exp.t
        if len(t) < 5:
            continue

        vx_b, vy_b = _world_to_body_xy(exp.vx, exp.vy, exp.yaw)
        dvx = _safe_gradient(vx_b, t)
        dvy = _safe_gradient(vy_b, t)

        # Body-frame identification (ux, uy are body commands).
        a1 = np.column_stack([exp.ux, vx_b, np.zeros_like(t), np.zeros_like(t)])
        a2 = np.column_stack([np.zeros_like(t), np.zeros_like(t), exp.uy, vy_b])

        a_blocks.append(a1)
        a_blocks.append(a2)
        b_blocks.append(dvx)
        b_blocks.append(dvy)

    if not a_blocks:
        raise RuntimeError("No XY data available to fit gamma1..gamma4")

    a = np.vstack(a_blocks)
    b = np.concatenate(b_blocks)

    theta, *_ = np.linalg.lstsq(a, b, rcond=None)
    pred = a @ theta
    rmse = float(np.sqrt(np.mean((pred - b) ** 2)))
    return theta, {
        "rmse_xy_accel": rmse,
        "n_samples_xy": int(len(b)),
        "xy_identification_frame": "body",
    }


def fit_1d(experiments: List[ExperimentData], axis: str) -> Tuple[np.ndarray, Dict[str, float]]:
    # axis 'z' -> [gamma5, gamma6], axis 'yaw' -> [gamma7, gamma8]
    a_blocks: List[np.ndarray] = []
    b_blocks: List[np.ndarray] = []

    for exp in experiments:
        if exp.axis != axis:
            continue

        t = exp.t
        if len(t) < 5:
            continue

        if axis == "z":
            dv = _safe_gradient(exp.vz, t)
            a = np.column_stack([exp.uz, exp.vz])
        else:
            dv = _safe_gradient(exp.wyaw, t)
            a = np.column_stack([exp.uyaw, exp.wyaw])

        a_blocks.append(a)
        b_blocks.append(dv)

    if not a_blocks:
        raise RuntimeError(f"No {axis} data available to fit parameters")

    a_all = np.vstack(a_blocks)
    b_all = np.concatenate(b_blocks)

    theta, *_ = np.linalg.lstsq(a_all, b_all, rcond=None)
    pred = a_all @ theta
    rmse = float(np.sqrt(np.mean((pred - b_all) ** 2)))
    return theta, {f"rmse_{axis}_accel": rmse, f"n_samples_{axis}": int(len(b_all))}


def simulate_experiment(exp: ExperimentData, gammas: Dict[str, float]) -> Dict[str, np.ndarray]:
    t = exp.t
    n = len(t)

    x_m = exp.x
    y_m = exp.y
    z_m = exp.z
    yaw_m = exp.yaw
    vx_m = exp.vx
    vy_m = exp.vy
    vz_m = exp.vz
    wyaw_m = exp.wyaw

    # Initialize model state at first measured sample.
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    yaw = np.zeros(n, dtype=float)
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    vxb = np.zeros(n, dtype=float)
    vyb = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)
    wyaw = np.zeros(n, dtype=float)

    x[0], y[0], z[0], yaw[0] = x_m[0], y_m[0], z_m[0], yaw_m[0]
    vx[0], vy[0], vz[0], wyaw[0] = vx_m[0], vy_m[0], vz_m[0], wyaw_m[0]
    vxb0, vyb0 = _world_to_body_xy(
        np.asarray([vx_m[0]], dtype=float),
        np.asarray([vy_m[0]], dtype=float),
        np.asarray([yaw_m[0]], dtype=float),
    )
    vxb[0] = float(vxb0[0])
    vyb[0] = float(vyb0[0])

    g1 = gammas["gamma1"]
    g2 = gammas["gamma2"]
    g3 = gammas["gamma3"]
    g4 = gammas["gamma4"]
    g5 = gammas["gamma5"]
    g6 = gammas["gamma6"]
    g7 = gammas["gamma7"]
    g8 = gammas["gamma8"]

    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-3

        ux = exp.ux[k]
        uy = exp.uy[k]
        uz = exp.uz[k]
        uyaw = exp.uyaw[k]

        # XY identification is done in body frame (vxb/vyb), so validation
        # simulation propagates body velocities and maps to world with measured yaw.
        axb = g2 * vxb[k] + g1 * ux
        ayb = g4 * vyb[k] + g3 * uy
        az = g6 * vz[k] + g5 * uz
        aw = g8 * wyaw[k] + g7 * uyaw

        vxb[k + 1] = vxb[k] + axb * dt
        vyb[k + 1] = vyb[k] + ayb * dt
        vz[k + 1] = vz[k] + az * dt
        wyaw[k + 1] = wyaw[k] + aw * dt

        psi_next = yaw_m[k + 1]
        c = math.cos(psi_next)
        s = math.sin(psi_next)
        vx[k + 1] = c * vxb[k + 1] - s * vyb[k + 1]
        vy[k + 1] = s * vxb[k + 1] + c * vyb[k + 1]

        x[k + 1] = x[k] + vx[k + 1] * dt
        y[k + 1] = y[k] + vy[k + 1] * dt
        z[k + 1] = z[k] + vz[k + 1] * dt
        yaw[k + 1] = yaw[k] + wyaw[k + 1] * dt

    yaw = _wrap_pi(yaw)

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "vx": vx,
        "vy": vy,
        "vxb": vxb,
        "vyb": vyb,
        "vz": vz,
        "wyaw": wyaw,
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def save_comparison_plot_and_csv(exp: ExperimentData, sim: Dict[str, np.ndarray], out_dir: Path) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t = exp.t - exp.t[0]

    vel_meas = [exp.vx, exp.vy, exp.vz, exp.wyaw]
    vel_mod = [sim["vx"], sim["vy"], sim["vz"], sim["wyaw"]]
    pos_meas = [exp.x, exp.y, exp.z, exp.yaw]
    pos_mod = [sim["x"], sim["y"], sim["z"], sim["yaw"]]

    vel_names = ["vx", "vy", "vz", "wyaw"]
    pos_names = ["x", "y", "z", "yaw"]

    cmd_signals = [exp.ux, exp.uy, exp.uz, exp.uyaw]
    cmd_names = ["ux_cmd", "uy_cmd", "uz_cmd", "uyaw_cmd"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)

    metrics: Dict[str, float] = {}

    for i in range(4):
        ax = axes[0, i]
        ax.plot(t, vel_meas[i], label="measured", linewidth=1.3)
        ax.plot(t, vel_mod[i], label="model", linewidth=1.3)
        ax.set_title(vel_names[i])
        ax.grid(True, alpha=0.3)
        metrics[f"rmse_{vel_names[i]}"] = _rmse(vel_meas[i], vel_mod[i])

    for i in range(4):
        ax = axes[1, i]
        ax.plot(t, pos_meas[i], label="measured", linewidth=1.3)
        ax.plot(t, pos_mod[i], label="model", linewidth=1.3)
        ax.set_title(pos_names[i])
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
        metrics[f"rmse_{pos_names[i]}"] = _rmse(pos_meas[i], pos_mod[i])

    for i in range(4):
        ax = axes[2, i]
        ax.plot(t, cmd_signals[i], color="tab:purple", linewidth=1.3)
        ax.set_title(cmd_names[i])
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{exp.name}: measured vs model")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = out_dir / f"{exp.name}_model_vs_measured.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    csv_path = out_dir / f"{exp.name}_model_vs_measured.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "x_meas",
                "x_model",
                "y_meas",
                "y_model",
                "z_meas",
                "z_model",
                "yaw_meas",
                "yaw_model",
                "vx_meas",
                "vx_model",
                "vy_meas",
                "vy_model",
                "vz_meas",
                "vz_model",
                "wyaw_meas",
                "wyaw_model",
                "ux",
                "uy",
                "uz",
                "uyaw",
            ]
        )
        for k in range(len(t)):
            writer.writerow(
                [
                    float(t[k]),
                    float(exp.x[k]),
                    float(sim["x"][k]),
                    float(exp.y[k]),
                    float(sim["y"][k]),
                    float(exp.z[k]),
                    float(sim["z"][k]),
                    float(exp.yaw[k]),
                    float(sim["yaw"][k]),
                    float(exp.vx[k]),
                    float(sim["vx"][k]),
                    float(exp.vy[k]),
                    float(sim["vy"][k]),
                    float(exp.vz[k]),
                    float(sim["vz"][k]),
                    float(exp.wyaw[k]),
                    float(sim["wyaw"][k]),
                    float(exp.ux[k]),
                    float(exp.uy[k]),
                    float(exp.uz[k]),
                    float(exp.uyaw[k]),
                ]
            )

    metrics["plot_path"] = str(plot_path)
    metrics["csv_path"] = str(csv_path)
    return metrics


def identify(root: Path, out_dir: Optional[Path],
             active_threshold: float, pad_s: float) -> None:
    if out_dir is None:
        out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_all_experiments(root, active_threshold=active_threshold, pad_s=pad_s)
    if not experiments:
        raise RuntimeError(f"No valid experiments found in: {root}")

    xy_theta, xy_stats = fit_xy(experiments)
    z_theta, z_stats = fit_1d(experiments, axis="z")
    yaw_theta, yaw_stats = fit_1d(experiments, axis="yaw")

    gammas = {
        "gamma1": float(xy_theta[0]),
        "gamma2": float(xy_theta[1]),
        "gamma3": float(xy_theta[2]),
        "gamma4": float(xy_theta[3]),
        "gamma5": float(z_theta[0]),
        "gamma6": float(z_theta[1]),
        "gamma7": float(yaw_theta[0]),
        "gamma8": float(yaw_theta[1]),
    }

    plots_dir = out_dir / "plots"
    per_exp_metrics: Dict[str, Dict[str, float]] = {}
    for exp in experiments:
        sim = simulate_experiment(exp, gammas)
        per_exp_metrics[exp.name] = save_comparison_plot_and_csv(exp, sim, plots_dir)

    result_yaml = {
        "root": str(root),
        "settings": {
            "use_measured_psi_in_validation_sim": True,
            "xy_identification_frame": "body",
            "active_threshold": float(active_threshold),
            "trim_pad_seconds": float(pad_s),
        },
        "gammas": gammas,
        "fit_stats": {
            **xy_stats,
            **z_stats,
            **yaw_stats,
            "n_experiments": len(experiments),
        },
        "experiments": [
            {
                "name": e.name,
                "axis": e.axis,
                "axis_source": e.axis_source,
                "samples": int(len(e.t)),
            }
            for e in experiments
        ],
        "comparison_metrics": per_exp_metrics,
    }

    yaml_path = out_dir / "identified_gammas.yaml"
    yaml_path.write_text(yaml.safe_dump(result_yaml, sort_keys=False))

    summary_lines = []
    summary_lines.append("Drone plant identification summary")
    summary_lines.append(f"root: {root}")
    summary_lines.append("")
    summary_lines.append("Estimated gammas:")
    for k in ["gamma1", "gamma2", "gamma3", "gamma4", "gamma5", "gamma6", "gamma7", "gamma8"]:
        summary_lines.append(f"  {k}: {gammas[k]:+.8f}")
    summary_lines.append("")
    summary_lines.append("Fit stats:")
    for k, v in {**xy_stats, **z_stats, **yaw_stats}.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append(f"identified_gammas.yaml: {yaml_path}")
    summary_lines.append(f"plots dir: {plots_dir}")

    txt_path = out_dir / "identification_summary.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n")

    print("Identification finished.")
    print(f"identified_gammas: {yaml_path}")
    print(f"summary: {txt_path}")
    print(f"plots: {plots_dir}")
    print("Estimated gammas:")
    for k in ["gamma1", "gamma2", "gamma3", "gamma4", "gamma5", "gamma6", "gamma7", "gamma8"]:
        print(f"  {k} = {gammas[k]:+.8f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify gamma1..gamma8 from drone experiments.")
    parser.add_argument(
        "--root",
        default="src/tello_control_pkg/experiments_ident",
        help="Root directory containing experiment folders with tello_interface CSV logs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for identified_gammas.yaml and plots (default: --root).",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-3,
        help="Threshold on active axis command magnitude to trim experiment window.",
    )
    parser.add_argument(
        "--trim-pad-s",
        type=float,
        default=1.0,
        help="Seconds padded before/after active command window.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    identify(
        root=root,
        out_dir=out_dir,
        active_threshold=float(args.active_threshold),
        pad_s=float(args.trim_pad_s),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
