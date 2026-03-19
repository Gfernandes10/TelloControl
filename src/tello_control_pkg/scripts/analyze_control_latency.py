#!/usr/bin/env python3
"""Analyze control/localization delays from CSV logs.

Reads, when available:
  - <log_dir>/tello_interface/filtered_pose.csv
  - <log_dir>/tello_interface/reference.csv
  - <log_dir>/tello_interface/u_control.csv
  - <log_dir>/tello_control_pkg/lqi_debug.csv
  - <log_dir>/localization_pkg/ground_truth.csv
  - <log_dir>/localization_pkg/filtered_pose.csv

Main outputs:
  - Pose stream jitter/gap statistics
  - Controller/manual command mix statistics
  - LQI loop dt statistics (from lqi_debug.csv)
  - Estimated command->response lag on all axes:
      ux->dx, uy->dy, uz->dz, uyaw->wyaw
  - Estimated localization lag on available axes (x/y/z/yaw), when available
  - Health checklist (pass/fail) for fast log triage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TimeSeries:
    t: np.ndarray
    v: np.ndarray


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _col_float(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    out: List[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=float)


def _col_str(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([str(row.get(key, "")) for row in rows], dtype=object)


def _clean_series(t: np.ndarray, v: np.ndarray) -> TimeSeries:
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    if len(t) == 0:
        return TimeSeries(np.zeros((0,), dtype=float), np.zeros((0,), dtype=float))
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    # Keep first sample for duplicate timestamps
    t_unique, idx = np.unique(t, return_index=True)
    return TimeSeries(t=t_unique, v=v[idx])


def _latest_log_dir(csv_logs_dir: Path) -> Path:
    candidates = [p for p in csv_logs_dir.iterdir() if p.is_dir() and p.name.startswith("20")]
    if not candidates:
        raise FileNotFoundError(f"No timestamped log directories in: {csv_logs_dir}")
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def _dt_stats(t: np.ndarray) -> Dict[str, float]:
    if len(t) < 2:
        return {
            "count": int(len(t)),
            "dt_mean": float("nan"),
            "dt_median": float("nan"),
            "dt_p95": float("nan"),
            "dt_max": float("nan"),
            "gaps_gt_0p1": 0,
            "gaps_gt_0p2": 0,
            "gaps_gt_0p5": 0,
        }
    dt = np.diff(np.sort(t))
    return {
        "count": int(len(t)),
        "dt_mean": float(np.mean(dt)),
        "dt_median": float(np.median(dt)),
        "dt_p95": float(np.quantile(dt, 0.95)),
        "dt_max": float(np.max(dt)),
        "gaps_gt_0p1": int(np.sum(dt > 0.1)),
        "gaps_gt_0p2": int(np.sum(dt > 0.2)),
        "gaps_gt_0p5": int(np.sum(dt > 0.5)),
    }


def _zoh_resample(times: np.ndarray, values: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if len(times) == 0 or len(values) == 0 or len(query_t) == 0:
        return np.zeros((len(query_t),), dtype=float)
    out = np.zeros((len(query_t),), dtype=float)
    j = 0
    for i, tq in enumerate(query_t):
        while j + 1 < len(times) and times[j + 1] <= tq:
            j += 1
        out[i] = values[j]
    return out


def _interp_resample(times: np.ndarray, values: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if len(times) == 0 or len(values) == 0 or len(query_t) == 0:
        return np.zeros((len(query_t),), dtype=float)
    return np.interp(query_t, times, values)


def _manual_window_mask(
    query_t: np.ndarray, manual_t: np.ndarray, half_window_s: float
) -> np.ndarray:
    if len(query_t) == 0:
        return np.zeros((0,), dtype=bool)
    if len(manual_t) == 0:
        return np.zeros_like(query_t, dtype=bool)
    mt = np.sort(manual_t)
    left = np.searchsorted(mt, query_t - half_window_s, side="left")
    right = np.searchsorted(mt, query_t + half_window_s, side="right")
    return right > left


def _xcorr_lag_seconds(
    x: np.ndarray, y: np.ndarray, dt: float, max_lag_s: float
) -> Dict[str, float]:
    if len(x) < 10 or len(y) < 10 or len(x) != len(y):
        return {"lag_s": float("nan"), "corr": float("nan")}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"lag_s": float("nan"), "corr": float("nan")}
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return {"lag_s": float("nan"), "corr": float("nan")}

    max_lag_n = max(1, int(round(max_lag_s / dt)))
    best_corr = -np.inf
    best_lag = 0
    for lag in range(-max_lag_n, max_lag_n + 1):
        if lag > 0:
            xw = x[:-lag]
            yw = y[lag:]
        elif lag < 0:
            xw = x[-lag:]
            yw = y[:lag]
        else:
            xw = x
            yw = y
        if len(xw) < 10:
            continue
        sxw = float(np.std(xw))
        syw = float(np.std(yw))
        if sxw < 1e-12 or syw < 1e-12:
            continue
        corr = float(np.dot(xw, yw) / (len(xw) * sxw * syw))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if not np.isfinite(best_corr):
        return {"lag_s": float("nan"), "corr": float("nan")}
    best_corr = max(-1.0, min(1.0, best_corr))
    return {"lag_s": float(best_lag * dt), "corr": float(best_corr)}


def _nearest_time_diff_stats(src_t: np.ndarray, ref_t: np.ndarray) -> Dict[str, float]:
    if len(src_t) == 0 or len(ref_t) == 0:
        return {"mean_s": float("nan"), "p95_s": float("nan"), "max_s": float("nan")}
    ref_t = np.sort(ref_t)
    diffs = np.zeros((len(src_t),), dtype=float)
    for i, t in enumerate(src_t):
        j = np.searchsorted(ref_t, t)
        candidates = []
        if 0 <= j < len(ref_t):
            candidates.append(abs(ref_t[j] - t))
        if j - 1 >= 0:
            candidates.append(abs(ref_t[j - 1] - t))
        diffs[i] = min(candidates) if candidates else float("nan")
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return {"mean_s": float("nan"), "p95_s": float("nan"), "max_s": float("nan")}
    return {
        "mean_s": float(np.mean(diffs)),
        "p95_s": float(np.quantile(diffs, 0.95)),
        "max_s": float(np.max(diffs)),
    }


def _yaw_rate_from_yaw_series(yaw_ts: TimeSeries) -> TimeSeries:
    if len(yaw_ts.t) < 3:
        return TimeSeries(
            t=yaw_ts.t.copy(),
            v=np.zeros((len(yaw_ts.t),), dtype=float),
        )
    yaw_unwrapped = np.unwrap(yaw_ts.v)
    wyaw = np.gradient(yaw_unwrapped, yaw_ts.t)
    return _clean_series(yaw_ts.t, wyaw)


def _estimate_axis_lag(
    u_ts: TimeSeries,
    y_ts: TimeSeries,
    manual_t: np.ndarray,
    manual_window_s: float,
    max_lag_s: float,
) -> Dict[str, object]:
    if len(u_ts.t) == 0 or len(y_ts.t) == 0:
        return {"status": "insufficient_data"}

    start = max(float(np.min(u_ts.t)), float(np.min(y_ts.t)))
    end = min(float(np.max(u_ts.t)), float(np.max(y_ts.t)))
    if not (start < end):
        return {"status": "insufficient_overlap"}

    y_dt = np.diff(y_ts.t)
    base_dt = float(np.median(y_dt)) if len(y_dt) else 0.01
    base_dt = max(0.005, min(0.05, base_dt))
    n = int(math.floor((end - start) / base_dt))
    if n < 50:
        return {"status": "insufficient_samples"}

    grid_t = start + np.arange(n, dtype=float) * base_dt
    u_grid = _zoh_resample(u_ts.t, u_ts.v, grid_t)
    y_grid = _interp_resample(y_ts.t, y_ts.v, grid_t)
    manual_mask = _manual_window_mask(grid_t, manual_t, manual_window_s)
    valid = ~manual_mask & np.isfinite(u_grid) & np.isfinite(y_grid)

    if np.any(valid):
        yv = y_grid[valid]
        q1, q99 = np.quantile(yv, [0.01, 0.99])
        valid = valid & (y_grid >= q1) & (y_grid <= q99)

    if int(np.sum(valid)) <= 50:
        return {
            "status": "insufficient_valid_samples",
            "grid_dt_s": base_dt,
            "samples": int(np.sum(valid)),
            "manual_excluded_samples": int(np.sum(manual_mask)),
        }

    lag = _xcorr_lag_seconds(u_grid[valid], y_grid[valid], base_dt, max_lag_s)
    if not np.isfinite(float(lag.get("corr", float("nan")))):
        return {
            "status": "insufficient_excitation",
            "grid_dt_s": base_dt,
            "samples": int(np.sum(valid)),
            "manual_excluded_samples": int(np.sum(manual_mask)),
        }
    return {
        "status": "ok",
        "grid_dt_s": base_dt,
        "samples": int(np.sum(valid)),
        "manual_excluded_samples": int(np.sum(manual_mask)),
        **lag,
    }


def _build_health_checks(report: Dict[str, object]) -> Dict[str, object]:
    checks: Dict[str, Dict[str, object]] = {}

    filtered_pose_dt = report.get("filtered_pose_dt")
    if isinstance(filtered_pose_dt, dict):
        gaps_02 = int(filtered_pose_dt.get("gaps_gt_0p2", 0))
        checks["pose_gaps_0p2"] = {
            "pass": gaps_02 <= 3,
            "value": gaps_02,
            "limit": 3,
            "desc": "filtered_pose gaps > 0.2s should be low",
        }

    manual_count = int(report.get("manual_command_count", 0))
    controller_count = int(report.get("controller_command_count", 0))
    total_cmd = manual_count + controller_count
    controller_share = (float(controller_count) / float(total_cmd)) if total_cmd > 0 else float("nan")
    checks["controller_share"] = {
        "pass": bool(np.isfinite(controller_share) and controller_share >= 0.8),
        "value": controller_share,
        "limit": 0.8,
        "desc": "controller command share should be >= 80%",
    }

    lqi_dt = report.get("lqi_debug_dt")
    if isinstance(lqi_dt, dict):
        p95_dt = float(lqi_dt.get("p95", float("nan")))
        checks["lqi_loop_dt_p95"] = {
            "pass": bool(np.isfinite(p95_dt) and p95_dt <= 0.03),
            "value": p95_dt,
            "limit": 0.03,
            "desc": "LQI dt p95 should be <= 30ms",
        }

    sat = report.get("lqi_debug_saturation")
    if isinstance(sat, dict):
        sat_z = float(sat.get("sat_cmd_z_pct", float("nan")))
        checks["sat_cmd_z_pct"] = {
            "pass": bool(np.isfinite(sat_z) and sat_z <= 5.0),
            "value": sat_z,
            "limit": 5.0,
            "desc": "Z command saturation should be <= 5%",
        }

    lag_keys = {
        "x": "control_response_lag_ux_to_dx",
        "y": "control_response_lag_uy_to_dy",
        "z": "control_response_lag_uz_to_dz",
        "yaw": "control_response_lag_uyaw_to_wyaw",
    }
    for axis, key in lag_keys.items():
        lag_obj = report.get(key)
        if not isinstance(lag_obj, dict) or lag_obj.get("status") != "ok":
            continue
        corr = float(lag_obj.get("corr", float("nan")))
        lag_s = float(lag_obj.get("lag_s", float("nan")))
        checks[f"lag_{axis}_corr"] = {
            "pass": bool(np.isfinite(corr) and corr >= 0.2),
            "value": corr,
            "limit": 0.2,
            "desc": f"cross-corr for axis {axis} should be >= 0.2",
        }
        checks[f"lag_{axis}_abs_s"] = {
            "pass": bool(np.isfinite(lag_s) and abs(lag_s) <= 1.5),
            "value": lag_s,
            "limit": 1.5,
            "desc": f"|lag| for axis {axis} should be <= 1.5s",
        }

    pass_count = int(sum(1 for item in checks.values() if bool(item.get("pass"))))
    total_count = len(checks)
    overall_pass = bool(total_count > 0 and pass_count == total_count)
    return {
        "overall_pass": overall_pass,
        "pass_count": pass_count,
        "total_count": total_count,
        "checks": checks,
    }


def analyze(log_dir: Path, manual_window_s: float, max_lag_s: float) -> Dict[str, object]:
    report: Dict[str, object] = {
        "log_dir": str(log_dir),
    }

    ti_dir = log_dir / "tello_interface"
    tc_dir = log_dir / "tello_control_pkg"
    loc_dir = log_dir / "localization_pkg"

    ref_path = ti_dir / "reference.csv"
    pose_path = ti_dir / "filtered_pose.csv"
    u_path = ti_dir / "u_control.csv"
    lqi_debug_path = tc_dir / "lqi_debug.csv"
    loc_gt_path = loc_dir / "ground_truth.csv"
    loc_filtered_path = loc_dir / "filtered_pose.csv"

    for p in [ref_path, pose_path, u_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    ref_rows = _read_csv_rows(ref_path)
    pose_rows = _read_csv_rows(pose_path)
    u_rows = _read_csv_rows(u_path)

    ref_t = _col_float(ref_rows, "timestamp")
    ref_x = _col_float(ref_rows, "x_ref")
    ref_y = _col_float(ref_rows, "y_ref")
    ref_z = _col_float(ref_rows, "z_ref")
    ref_yaw = _col_float(ref_rows, "yaw_ref")
    ref_xs = _clean_series(ref_t, ref_x)
    ref_ys = _clean_series(ref_t, ref_y)
    ref_zs = _clean_series(ref_t, ref_z)
    ref_yaws = _clean_series(ref_t, ref_yaw)

    pose_t = _col_float(pose_rows, "timestamp")
    pose_x = _col_float(pose_rows, "x")
    pose_dx = _col_float(pose_rows, "dx")
    pose_y = _col_float(pose_rows, "y")
    pose_dy = _col_float(pose_rows, "dy")
    pose_z = _col_float(pose_rows, "z")
    pose_dz = _col_float(pose_rows, "dz")
    pose_yaw = _col_float(pose_rows, "yaw")
    pose_r = _col_float(pose_rows, "r")
    pose_xs = _clean_series(pose_t, pose_x)
    pose_dxs = _clean_series(pose_t, pose_dx)
    pose_ys = _clean_series(pose_t, pose_y)
    pose_dys = _clean_series(pose_t, pose_dy)
    pose_zs = _clean_series(pose_t, pose_z)
    pose_dzs = _clean_series(pose_t, pose_dz)
    pose_yaws = _clean_series(pose_t, pose_yaw)
    pose_rs = _clean_series(pose_t, pose_r)
    if len(pose_rs.t) < 10:
        pose_rs = _yaw_rate_from_yaw_series(pose_yaws)

    u_t = _col_float(u_rows, "timestamp")
    u_ux = _col_float(u_rows, "ux")
    u_uy = _col_float(u_rows, "uy")
    u_uz = _col_float(u_rows, "uz")
    u_uyaw = _col_float(u_rows, "uyaw")
    u_source = _col_str(u_rows, "source")
    u_uxs = _clean_series(u_t, u_ux)
    u_uys = _clean_series(u_t, u_uy)
    u_uzs = _clean_series(u_t, u_uz)
    u_uyaws = _clean_series(u_t, u_uyaw)

    source_counts: Dict[str, int] = {}
    for src in u_source:
        source_counts[str(src)] = source_counts.get(str(src), 0) + 1
    report["u_source_counts"] = source_counts

    report["filtered_pose_dt"] = _dt_stats(pose_zs.t)
    report["reference_dt"] = _dt_stats(ref_zs.t)
    report["u_control_dt"] = _dt_stats(u_uzs.t)

    # Manual window exclusion defined from u_control source column.
    manual_t = u_t[(u_source == "manual") & np.isfinite(u_t)]
    controller_t = u_t[(u_source == "controller") & np.isfinite(u_t)]
    report["manual_command_count"] = int(len(manual_t))
    report["controller_command_count"] = int(len(controller_t))

    # LQI debug stats if available.
    if lqi_debug_path.exists():
        lqi_rows = _read_csv_rows(lqi_debug_path)
        lqi_t = _col_float(lqi_rows, "timestamp")
        lqi_dt = _col_float(lqi_rows, "dt")
        lqi_uz_raw = _col_float(lqi_rows, "v_raw_z")
        lqi_uz_pub = _col_float(lqi_rows, "cmd_published_z")
        sat_cmd_x = _col_float(lqi_rows, "sat_cmd_x")
        sat_cmd_y = _col_float(lqi_rows, "sat_cmd_y")
        sat_cmd_z = _col_float(lqi_rows, "sat_cmd_z")
        sat_cmd_yaw = _col_float(lqi_rows, "sat_cmd_yaw")
        sat_int_x = _col_float(lqi_rows, "sat_int_ex")
        sat_int_y = _col_float(lqi_rows, "sat_int_ey")
        sat_int_z = _col_float(lqi_rows, "sat_int_ez")
        sat_int_yaw = _col_float(lqi_rows, "sat_int_eyaw")
        aw_z_block = _col_float(lqi_rows, "aw_z_integrator_blocked")
        dt_clean = lqi_dt[np.isfinite(lqi_dt)]
        report["lqi_debug_dt"] = {
            "count": int(len(dt_clean)),
            "mean": float(np.mean(dt_clean)) if len(dt_clean) else float("nan"),
            "median": float(np.median(dt_clean)) if len(dt_clean) else float("nan"),
            "p95": float(np.quantile(dt_clean, 0.95)) if len(dt_clean) else float("nan"),
            "max": float(np.max(dt_clean)) if len(dt_clean) else float("nan"),
        }
        sat_cmd_x_mask = sat_cmd_x[np.isfinite(sat_cmd_x)]
        sat_cmd_y_mask = sat_cmd_y[np.isfinite(sat_cmd_y)]
        sat_cmd_z_mask = sat_cmd_z[np.isfinite(sat_cmd_z)]
        sat_cmd_yaw_mask = sat_cmd_yaw[np.isfinite(sat_cmd_yaw)]
        sat_int_x_mask = sat_int_x[np.isfinite(sat_int_x)]
        sat_int_y_mask = sat_int_y[np.isfinite(sat_int_y)]
        sat_int_z_mask = sat_int_z[np.isfinite(sat_int_z)]
        sat_int_yaw_mask = sat_int_yaw[np.isfinite(sat_int_yaw)]
        aw_z_block_mask = aw_z_block[np.isfinite(aw_z_block)]
        report["lqi_debug_saturation"] = {
            "sat_cmd_x_pct": float(np.mean(sat_cmd_x_mask > 0.5) * 100.0) if len(sat_cmd_x_mask) else float("nan"),
            "sat_cmd_y_pct": float(np.mean(sat_cmd_y_mask > 0.5) * 100.0) if len(sat_cmd_y_mask) else float("nan"),
            "sat_cmd_z_pct": float(np.mean(sat_cmd_z_mask > 0.5) * 100.0) if len(sat_cmd_z_mask) else float("nan"),
            "sat_cmd_yaw_pct": float(np.mean(sat_cmd_yaw_mask > 0.5) * 100.0) if len(sat_cmd_yaw_mask) else float("nan"),
            "sat_int_ex_pct": float(np.mean(sat_int_x_mask > 0.5) * 100.0) if len(sat_int_x_mask) else float("nan"),
            "sat_int_ey_pct": float(np.mean(sat_int_y_mask > 0.5) * 100.0) if len(sat_int_y_mask) else float("nan"),
            "sat_int_ez_pct": float(np.mean(sat_int_z_mask > 0.5) * 100.0) if len(sat_int_z_mask) else float("nan"),
            "sat_int_eyaw_pct": float(np.mean(sat_int_yaw_mask > 0.5) * 100.0) if len(sat_int_yaw_mask) else float("nan"),
            "aw_z_integrator_blocked_pct": float(np.mean(aw_z_block_mask > 0.5) * 100.0) if len(aw_z_block_mask) else float("nan"),
        }
        lqi_ts = lqi_t[np.isfinite(lqi_t)]
        report["lqi_to_u_time_diff"] = _nearest_time_diff_stats(lqi_ts, controller_t)
        # Keep a short sanity line for command pipeline
        report["lqi_cmd_vs_u_control"] = {
            "mean_abs_diff_uz": float(np.nanmean(np.abs(lqi_uz_pub - lqi_uz_raw)))
            if len(lqi_uz_pub) == len(lqi_uz_raw) and len(lqi_uz_pub) > 0 else float("nan")
        }
    else:
        report["lqi_debug_dt"] = "lqi_debug.csv not found"

    report["control_response_lag_ux_to_dx"] = _estimate_axis_lag(
        u_uxs, pose_dxs, manual_t, manual_window_s, max_lag_s
    )
    report["control_response_lag_uy_to_dy"] = _estimate_axis_lag(
        u_uys, pose_dys, manual_t, manual_window_s, max_lag_s
    )
    report["control_response_lag_uz_to_dz"] = _estimate_axis_lag(
        u_uzs, pose_dzs, manual_t, manual_window_s, max_lag_s
    )
    report["control_response_lag_uyaw_to_wyaw"] = _estimate_axis_lag(
        u_uyaws, pose_rs, manual_t, manual_window_s, max_lag_s
    )

    # Localization lag (raw ground truth -> filtered), when available.
    if loc_gt_path.exists() and loc_filtered_path.exists():
        gt_rows = _read_csv_rows(loc_gt_path)
        lf_rows = _read_csv_rows(loc_filtered_path)
        gt_t = _col_float(gt_rows, "timestamp")
        lf_t = _col_float(lf_rows, "timestamp")
        gt_xs = _clean_series(gt_t, _col_float(gt_rows, "x"))
        gt_ys = _clean_series(gt_t, _col_float(gt_rows, "y"))
        gt_zs = _clean_series(gt_t, _col_float(gt_rows, "z"))
        gt_yaws = _clean_series(gt_t, _col_float(gt_rows, "yaw"))
        lf_xs = _clean_series(lf_t, _col_float(lf_rows, "x"))
        lf_ys = _clean_series(lf_t, _col_float(lf_rows, "y"))
        lf_zs = _clean_series(lf_t, _col_float(lf_rows, "z"))
        lf_yaws = _clean_series(lf_t, _col_float(lf_rows, "yaw"))
        report["localization_ground_truth_dt"] = _dt_stats(gt_zs.t)
        report["localization_filtered_dt"] = _dt_stats(lf_zs.t)

        for axis, gt_axis, lf_axis in [
            ("x", gt_xs, lf_xs),
            ("y", gt_ys, lf_ys),
            ("z", gt_zs, lf_zs),
            ("yaw", gt_yaws, lf_yaws),
        ]:
            if len(gt_axis.t) == 0 or len(lf_axis.t) == 0:
                report[f"localization_lag_groundtruth_to_filtered_{axis}"] = {"status": "insufficient_data"}
                continue
            ls = max(float(np.min(gt_axis.t)), float(np.min(lf_axis.t)))
            le = min(float(np.max(gt_axis.t)), float(np.max(lf_axis.t)))
            if not (ls < le):
                report[f"localization_lag_groundtruth_to_filtered_{axis}"] = {"status": "insufficient_overlap"}
                continue
            dt = float(np.median(np.diff(lf_axis.t))) if len(lf_axis.t) > 1 else 0.02
            dt = max(0.005, min(0.05, dt))
            n = int(math.floor((le - ls) / dt))
            if n < 50:
                report[f"localization_lag_groundtruth_to_filtered_{axis}"] = {"status": "insufficient_samples"}
                continue
            grid_t = ls + np.arange(n, dtype=float) * dt
            gt_grid = _interp_resample(gt_axis.t, gt_axis.v, grid_t)
            lf_grid = _interp_resample(lf_axis.t, lf_axis.v, grid_t)
            lag_gt_to_filtered = _xcorr_lag_seconds(gt_grid, lf_grid, dt, max_lag_s)
            report[f"localization_lag_groundtruth_to_filtered_{axis}"] = {
                "status": "ok",
                "grid_dt_s": dt,
                "samples": int(len(grid_t)),
                **lag_gt_to_filtered,
            }

    def _ref_changes(ts: TimeSeries, eps: float) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        if len(ts.t) == 0:
            return out
        prev = float(ts.v[0])
        out.append((float(ts.t[0]), prev))
        for t, v in zip(ts.t[1:], ts.v[1:]):
            vf = float(v)
            if abs(vf - prev) > eps:
                out.append((float(t), vf))
                prev = vf
        return out

    report["reference_x_changes"] = _ref_changes(ref_xs, 1e-9)
    report["reference_y_changes"] = _ref_changes(ref_ys, 1e-9)
    report["reference_z_changes"] = _ref_changes(ref_zs, 1e-9)
    report["reference_yaw_changes"] = _ref_changes(ref_yaws, 1e-9)
    report["health_checks"] = _build_health_checks(report)

    return report


def _fmt_float(v: object, digits: int = 6) -> str:
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return "nan"
        return f"{float(v):.{digits}f}"
    return str(v)


def print_report(report: Dict[str, object]) -> None:
    print(f"[latency] log_dir: {report['log_dir']}")
    print(f"[latency] u_source_counts: {report.get('u_source_counts')}")
    print(f"[latency] manual_command_count: {report.get('manual_command_count')}")
    print(f"[latency] controller_command_count: {report.get('controller_command_count')}")

    for key in ["filtered_pose_dt", "reference_dt", "u_control_dt", "localization_filtered_dt", "localization_ground_truth_dt"]:
        if key in report:
            value = report[key]
            if isinstance(value, dict):
                print(
                    f"[latency] {key}: mean={_fmt_float(value.get('dt_mean'))} "
                    f"median={_fmt_float(value.get('dt_median'))} p95={_fmt_float(value.get('dt_p95'))} "
                    f"max={_fmt_float(value.get('dt_max'))} gaps>0.2={value.get('gaps_gt_0p2')}"
                )

    lqi_dt = report.get("lqi_debug_dt")
    if isinstance(lqi_dt, dict):
        print(
            f"[latency] lqi_debug_dt: mean={_fmt_float(lqi_dt.get('mean'))} "
            f"median={_fmt_float(lqi_dt.get('median'))} p95={_fmt_float(lqi_dt.get('p95'))} "
            f"max={_fmt_float(lqi_dt.get('max'))}"
        )
    else:
        print(f"[latency] lqi_debug_dt: {lqi_dt}")

    sat = report.get("lqi_debug_saturation")
    if isinstance(sat, dict):
        print(
            f"[latency] lqi saturation cmd[%]: "
            f"x={_fmt_float(sat.get('sat_cmd_x_pct'), 2)} "
            f"y={_fmt_float(sat.get('sat_cmd_y_pct'), 2)} "
            f"z={_fmt_float(sat.get('sat_cmd_z_pct'), 2)} "
            f"yaw={_fmt_float(sat.get('sat_cmd_yaw_pct'), 2)}"
        )
        print(
            f"[latency] lqi saturation int[%]: "
            f"ex={_fmt_float(sat.get('sat_int_ex_pct'), 2)} "
            f"ey={_fmt_float(sat.get('sat_int_ey_pct'), 2)} "
            f"ez={_fmt_float(sat.get('sat_int_ez_pct'), 2)} "
            f"eyaw={_fmt_float(sat.get('sat_int_eyaw_pct'), 2)} "
            f"aw_z_block={_fmt_float(sat.get('aw_z_integrator_blocked_pct'), 2)}"
        )

    td = report.get("lqi_to_u_time_diff")
    if isinstance(td, dict):
        print(
            f"[latency] lqi->u nearest time diff: mean={_fmt_float(td.get('mean_s'))}s "
            f"p95={_fmt_float(td.get('p95_s'))}s max={_fmt_float(td.get('max_s'))}s"
        )

    lag_items = [
        ("ux->dx", "control_response_lag_ux_to_dx"),
        ("uy->dy", "control_response_lag_uy_to_dy"),
        ("uz->dz", "control_response_lag_uz_to_dz"),
        ("uyaw->wyaw", "control_response_lag_uyaw_to_wyaw"),
    ]
    for label, key in lag_items:
        c2r = report.get(key)
        if isinstance(c2r, dict) and c2r.get("status") == "ok":
            print(
                f"[latency] control->response ({label}): lag={_fmt_float(c2r.get('lag_s'))}s "
                f"corr={_fmt_float(c2r.get('corr'), 4)} samples={c2r.get('samples')} "
                f"(manual_excluded={c2r.get('manual_excluded_samples')})"
            )
        else:
            print(f"[latency] control->response ({label}): {c2r}")

    for axis in ("x", "y", "z", "yaw"):
        key = f"localization_lag_groundtruth_to_filtered_{axis}"
        loc = report.get(key)
        if isinstance(loc, dict) and loc.get("status") == "ok":
            print(
                f"[latency] localization lag (gt {axis} -> filtered {axis}): "
                f"lag={_fmt_float(loc.get('lag_s'))}s corr={_fmt_float(loc.get('corr'), 4)} "
                f"samples={loc.get('samples')}"
            )
        elif loc is not None:
            print(f"[latency] localization lag (gt {axis} -> filtered {axis}): {loc}")

    for axis in ("x", "y", "z", "yaw"):
        changes = report.get(f"reference_{axis}_changes")
        if isinstance(changes, list) and len(changes):
            short = ", ".join([f"({t:.3f}->{v:.3f})" for t, v in changes[:6]])
            tail = " ..." if len(changes) > 6 else ""
            print(f"[latency] reference {axis} changes: {short}{tail}")

    health = report.get("health_checks")
    if isinstance(health, dict):
        print(
            f"[latency] health overall_pass={health.get('overall_pass')} "
            f"({health.get('pass_count')}/{health.get('total_count')})"
        )
        checks = health.get("checks", {})
        if isinstance(checks, dict):
            failed = [name for name, item in checks.items() if isinstance(item, dict) and not bool(item.get("pass"))]
            if failed:
                print(f"[latency] health failed_checks: {', '.join(failed)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze control/localization delays from CSV logs.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Path to one timestamped log dir (e.g. csvLogs/20260301_1626). Default: latest in --csv-logs-dir.",
    )
    parser.add_argument(
        "--csv-logs-dir",
        type=Path,
        default=Path("csvLogs"),
        help="Parent logs directory used when --log-dir is not provided.",
    )
    parser.add_argument(
        "--manual-window",
        type=float,
        default=0.05,
        help="Seconds around each manual command to exclude from controller-only analysis.",
    )
    parser.add_argument(
        "--max-lag",
        type=float,
        default=3.0,
        help="Max lag window (s) for cross-correlation-based lag estimation.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save JSON report. Default: <log_dir>/tello_control_pkg/latency_report.json",
    )
    args = parser.parse_args()

    if args.log_dir is None:
        log_dir = _latest_log_dir(args.csv_logs_dir)
    else:
        log_dir = args.log_dir

    report = analyze(log_dir, manual_window_s=args.manual_window, max_lag_s=args.max_lag)
    print_report(report)

    out_path = args.save_json
    if out_path is None:
        out_dir = log_dir / "tello_control_pkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latency_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[latency] saved json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
