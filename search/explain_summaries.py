import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HC_PATH = Path("../eval_outputs/hc_summary.json")
RS_PATH = Path("../eval_outputs/random_eval_summary.json")


def load(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


hc = load(HC_PATH)
rs = load(RS_PATH)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_summary(s: dict) -> dict:
    # Failure discovery
    found = safe_get(s, "failure_discovery", "found_collision")
    num_crashes = safe_get(s, "failure_discovery", "num_crashes")
    first_it = safe_get(s, "failure_discovery", "first_crash_iteration")
    success_rate = safe_get(s, "failure_discovery", "success_rate")

    # Scenario characteristics
    distinct = safe_get(s, "scenario_characteristics", "num_distinct_crash_configs")
    most_prob = safe_get(s, "scenario_characteristics", "most_problematic_config", default={})

    # Efficiency
    runtime = safe_get(s, "efficiency", "runtime_s")
    total_eval = safe_get(s, "efficiency", "total_evaluations")
    it_to_first = safe_get(s, "efficiency", "iterations_to_first_crash")

    # Diagnostics
    avg_md = safe_get(s, "diagnostics", "avg_min_distance_no_crash")
    min_md = safe_get(s, "diagnostics", "min_distance_no_crash")
    min_fit = safe_get(s, "diagnostics", "min_fitness_no_crash")
    avg_fit = safe_get(s, "diagnostics", "avg_fitness_no_crash")

    return {
        "method": s.get("method"),
        "n_scenarios": s.get("n_scenarios"),
        "found_collision": found,
        "num_crashes": num_crashes,
        "success_rate": success_rate,
        "first_crash_iteration": first_it,
        "iterations_to_first_crash": it_to_first,
        "total_evaluations": total_eval,
        "runtime_s": runtime,
        "evals_per_sec": (total_eval / runtime) if (runtime and total_eval) else None,
        "num_distinct_crash_configs": distinct,
        "avg_min_distance_no_crash": avg_md,
        "min_distance_no_crash": min_md,
        "min_fitness_no_crash": min_fit,
        "avg_fitness_no_crash": avg_fit,
        # pull a few key "most problematic" params for comparison plots
        "most_prob_lanes_count": most_prob.get("lanes_count"),
        "most_prob_vehicles_count": most_prob.get("vehicles_count"),
        "most_prob_initial_spacing": most_prob.get("initial_spacing"),
        "most_prob_ego_spacing": most_prob.get("ego_spacing"),
        "most_prob_initial_lane_id": most_prob.get("initial_lane_id"),
    }


# ===============================
# Base summary table (keep this)
# ===============================
df = pd.DataFrame([flatten_summary(hc), flatten_summary(rs)]).set_index("method")
print("\n=== Summary table ===\n")
print(df[[
    "n_scenarios", "found_collision", "num_crashes", "num_distinct_crash_configs",
    "first_crash_iteration", "total_evaluations", "runtime_s", "evals_per_sec",
    "min_distance_no_crash", "avg_min_distance_no_crash", "min_fitness_no_crash"
]].to_string())


# -------------------------------
# Output dir
# -------------------------------
out_dir = Path("analysis_plots")
out_dir.mkdir(exist_ok=True)

