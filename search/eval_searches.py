import os
import json
import csv
import time
import copy
from collections import defaultdict, Counter

import numpy as np
from numpy import mean

from search.random_search import RandomSearch
from search.hill_climbing import compute_objectives_from_time_series, compute_fitness, hill_climb


# from typing import Dict, Any, Optional, List, Tuple
#
# # We reuse the same objective + fitness computation as your HC implementation
# # so random vs HC are comparable.
# from search import hill_climbing as hc_mod  # provides compute_objectives_from_time_series, compute_fitness
#
#
# # -------------------------
# # Global helpers
# # -------------------------
# def _ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)
#
#
# def _flatten(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
#     return {f"{prefix}.{k}": v for k, v in d.items()}
#
#
# def _scenario_key(cfg: Dict[str, Any]) -> Tuple:
#     keys = ["vehicles_count", "lanes_count", "initial_spacing", "ego_spacing", "initial_lane_id"]
#     return tuple(cfg.get(k, None) for k in keys)
#
#
# def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(rec) + "\n")
#
#
# def _append_csv(path: str, rec: Dict[str, Any], header: Optional[List[str]]) -> List[str]:
#     file_exists = os.path.exists(path)
#
#     # Load header if file exists
#     if file_exists and header is None:
#         with open(path, "r", encoding="utf-8", newline="") as f:
#             reader = csv.reader(f)
#             header = next(reader, None)
#
#     if header is None:
#         header = list(rec.keys())
#
#     # Expand header if new keys appear
#     missing = [k for k in rec.keys() if k not in header]
#     if missing:
#         header = header + missing
#
#     with open(path, "a", encoding="utf-8", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=header)
#         if not file_exists:
#             w.writeheader()
#         w.writerow({k: rec.get(k, None) for k in header})
#
#     return header
#
#
# # -------------------------
# # Core patching wrapper
# # -------------------------
# def _make_logged_run_episode(real_run_episode, method_name: str, state: Dict[str, Any]):
#     """
#     Wrap envs.highway_env_utils.run_episode (or whatever function is imported as run_episode
#     in the target module) to log every evaluation.
#
#     state holds:
#       - jsonl_path, csv_path, csv_header
#       - eval_idx, best, best_eval_idx, first_collision_eval_idx
#       - distinct_failing set, best_so_far list
#     """
#
#     def _wrapped(env_id: str, cfg: Dict[str, Any], policy, defaults: Dict[str, Any], seed_base: int):
#         t0 = time.time()
#         crashed_env, ts = real_run_episode(env_id, cfg, policy, defaults, seed_base)
#         t1 = time.time()
#
#         objectives = hc_mod.compute_objectives_from_time_series(ts)
#         fitness = float(hc_mod.compute_fitness(objectives))
#         crashed = bool(crashed_env) or bool(objectives.get("crashed", 0))
#
#         # failure bookkeeping
#         if crashed:
#             state["distinct_failing"].add(_scenario_key(cfg))
#             if state["first_collision_eval_idx"] is None:
#                 state["first_collision_eval_idx"] = state["eval_idx"]
#
#         # logging record
#         rec = {
#             "method": method_name,
#             "eval_idx": state["eval_idx"],
#             "eval_time_s": (t1 - t0),
#             "seed_base": int(seed_base),
#             "crashed": crashed,
#             "fitness": fitness,
#             "min_distance": objectives.get("min_euclidean_distance", None),
#             **_flatten("cfg", cfg),
#             **_flatten("obj", objectives),
#         }
#         _append_jsonl(state["jsonl_path"], rec)
#         state["csv_header"] = _append_csv(state["csv_path"], rec, state["csv_header"])
#
#         # best definition: crash dominates; else lower fitness is better
#         cand = {
#             "crashed": crashed,
#             "fitness": fitness,
#             "cfg": copy.deepcopy(cfg),
#             "objectives": objectives,
#             "seed_base": int(seed_base),
#             "eval_idx": int(state["eval_idx"]),
#         }
#         best = state["best"]
#         if best is None:
#             state["best"] = cand
#             state["best_eval_idx"] = state["eval_idx"]
#         else:
#             if cand["crashed"] and not best["crashed"]:
#                 state["best"] = cand
#                 state["best_eval_idx"] = state["eval_idx"]
#             elif cand["crashed"] == best["crashed"] and cand["fitness"] < best["fitness"]:
#                 state["best"] = cand
#                 state["best_eval_idx"] = state["eval_idx"]
#
#         state["best_so_far"].append(state["best"]["fitness"])
#         state["eval_idx"] += 1
#
#         return crashed_env, ts
#
#     return _wrapped
#
#
# def _build_summary(method: str, state: Dict[str, Any], runtime_s: float, extra: Dict[str, Any]) -> Dict[str, Any]:
#     best = state["best"]
#     found_collision = bool(best["crashed"]) if best else False
#     best_min_dist = best["objectives"].get("min_euclidean_distance", None) if best else None
#
#     if found_collision and state["first_collision_eval_idx"] is not None:
#         evals_to_target = state["first_collision_eval_idx"] + 1
#     else:
#         evals_to_target = (state["best_eval_idx"] + 1) if state["best_eval_idx"] is not None else None
#
#     most_cfg = best["cfg"] if best else None
#
#     summary = {
#         "method": method,
#
#         # 1) Failure discovery
#         "failure_discovery": {
#             "found_collision": found_collision,
#             "distinct_failing_scenarios": len(state["distinct_failing"]),
#             "min_distance_if_no_collision": (best_min_dist if not found_collision else None),
#         },
#
#         # 2) Scenario characteristics
#         "scenario_characteristics": {
#             "most_critical_cfg": most_cfg,
#             "most_critical_min_distance": best_min_dist,
#             "most_critical_env_config": (
#                 {
#                     "vehicles_count": most_cfg.get("vehicles_count"),
#                     "lanes_count": most_cfg.get("lanes_count"),
#                     "initial_spacing": most_cfg.get("initial_spacing"),
#                     "ego_spacing": most_cfg.get("ego_spacing"),
#                     "initial_lane_id": most_cfg.get("initial_lane_id"),
#                 } if most_cfg else None
#             ),
#         },
#
#         # 3) Efficiency
#         "efficiency": {
#             "runtime_s": float(runtime_s),
#             "total_evaluations": int(state["eval_idx"]),
#             "evaluations_to_collision_or_closest": evals_to_target,
#             "best_eval_idx": state["best_eval_idx"],
#             "first_collision_eval_idx": state["first_collision_eval_idx"],
#             "best_fitness": (best["fitness"] if best else None),
#         },
#
#         # helpful for scalability plots (best fitness vs evaluations)
#         "best_so_far_fitness_curve": state["best_so_far"],
#
#         "paths": {
#             "jsonl": state["jsonl_path"],
#             "csv": state["csv_path"],
#             "summary": state["summary_path"],
#         },
#
#         **extra,
#     }
#
#     with open(state["summary_path"], "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)
#
#     return summary
#
#
# # ==========================================================
# # 1) Evaluate Random Search (uses RandomSearch.run_search)
# # ==========================================================
# def eval_random_search(
#         env_id: str,
#         base_cfg: Dict[str, Any],
#         param_spec: Dict[str, Any],
#         policy,
#         defaults: Dict[str, Any],
#         *,
#         n_scenarios: int = 50,
#         n_eval: int = 1,
#         seed: int = 11,
#         out_dir: str = "eval_outputs/random",
#         save_videos: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Runs the existing RandomSearch.run_search and logs every evaluation + writes summary stats.
#     """
#     from search.random_search import RandomSearch
#     import search.random_search as rs_mod  # patch where run_search resolves run_episode
#
#     _ensure_dir(out_dir)
#     state = {
#         "jsonl_path": os.path.join(out_dir, "random_evals.jsonl"),
#         "csv_path": os.path.join(out_dir, "random_evals.csv"),
#         "summary_path": os.path.join(out_dir, "random_summary.json"),
#         "csv_header": None,
#
#         "eval_idx": 0,
#         "distinct_failing": set(),
#         "first_collision_eval_idx": None,
#
#         "best": None,
#         "best_eval_idx": None,
#         "best_so_far": [],
#     }
#
#     # create search object (as in your template)
#     search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
#
#     # patch module-level functions used by run_search
#     orig_run_episode = rs_mod.run_episode
#     orig_record_video = getattr(rs_mod, "record_video_episode", None)
#
#     rs_mod.run_episode = _make_logged_run_episode(orig_run_episode, "random", state)
#     if orig_record_video is not None and not save_videos:
#         rs_mod.record_video_episode = lambda *args, **kwargs: None
#
#     t0 = time.time()
#     try:
#         run_search_return = search.run_search(n_scenarios=n_scenarios, n_eval=n_eval, seed=seed)
#     finally:
#         rs_mod.run_episode = orig_run_episode
#         if orig_record_video is not None:
#             rs_mod.record_video_episode = orig_record_video
#     t1 = time.time()
#
#     return _build_summary(
#         method="random",
#         state=state,
#         runtime_s=(t1 - t0),
#         extra={
#             "seed": seed,
#             "n_scenarios": n_scenarios,
#             "n_eval": n_eval,
#             "run_search_return": run_search_return,
#         },
#     )
#
#
# # ==========================================================
# # 2) Evaluate Hill Climbing (uses hill_climb function)
# # ==========================================================
# def eval_hill_climb(
#         env_id: str,
#         base_cfg: Dict[str, Any],
#         param_spec: Dict[str, Any],
#         policy,
#         defaults: Dict[str, Any],
#         *,
#         neighbors_per_iter: int = 5,
#         iterations: int = 10,
#         seed: int = 9653893457,
#         out_dir: str = "eval_outputs/hc",
#         save_videos: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Runs the existing hill_climb(...) and logs every evaluation + writes summary stats.
#     """
#     import search.hill_climbing as hc_impl  # patch where hill_climb resolves run_episode
#     from search.hill_climbing import hill_climb
#
#     _ensure_dir(out_dir)
#     state = {
#         "jsonl_path": os.path.join(out_dir, "hc_evals.jsonl"),
#         "csv_path": os.path.join(out_dir, "hc_evals.csv"),
#         "summary_path": os.path.join(out_dir, "hc_summary.json"),
#         "csv_header": None,
#
#         "eval_idx": 0,
#         "distinct_failing": set(),
#         "first_collision_eval_idx": None,
#
#         "best": None,
#         "best_eval_idx": None,
#         "best_so_far": [],
#     }
#
#     # patch module-level functions used by hill_climb
#     orig_run_episode = hc_impl.run_episode
#     orig_record_video = getattr(hc_impl, "record_video_episode", None)
#
#     hc_impl.run_episode = _make_logged_run_episode(orig_run_episode, "hc", state)
#     if orig_record_video is not None and not save_videos:
#         hc_impl.record_video_episode = lambda *args, **kwargs: None
#
#     t0 = time.time()
#     try:
#         hc_return = hill_climb(
#             env_id, base_cfg, param_spec, policy, defaults,
#             neighbors_per_iter=neighbors_per_iter,
#             iterations=iterations,
#             seed=seed
#         )
#     finally:
#         hc_impl.run_episode = orig_run_episode
#         if orig_record_video is not None:
#             hc_impl.record_video_episode = orig_record_video
#     t1 = time.time()
#
#     return _build_summary(
#         method="hc",
#         state=state,
#         runtime_s=(t1 - t0),
#         extra={
#             "seed": seed,
#             "neighbors_per_iter": neighbors_per_iter,
#             "iterations": iterations,
#             "hill_climb_return": hc_return,
#         },
#     )
#
#
# def eval_hc_until_budget(
#         env_id,
#         base_cfg,
#         param_spec,
#         policy,
#         defaults,
#         *,
#         hill_climb_fn,
#         hill_climbing_module,
#         make_logged_run_episode,
#         state,
#         target_evals: int,
#         neighbors_per_iter: int,
#         iterations: int,
#         base_seed: int = 0,
#         out_dir: str = "eval_outputs/hc_budget",
# ):
#
#     os.makedirs(out_dir, exist_ok=True)
#     runs = []
#     crashes_found_runs = 0  # run-level success count (>=1 crash in that hill_climb run)
#     crash_eval_indices = []  # eval indices where crashes happened (from state, if you log them)
#
#     # Patch HC's run_episode so every evaluation is logged into state
#     orig_run_episode = hill_climbing_module.run_episode
#     orig_record_video = getattr(hill_climbing_module, "record_video_episode", None)
#
#     hill_climbing_module.run_episode = make_logged_run_episode(orig_run_episode, "hc", state)
#     if orig_record_video is not None:
#         hill_climbing_module.record_video_episode = lambda *args, **kwargs: None
#
#     t0 = time.time()
#     try:
#         run_id = 0
#         # continue until budget met/exceeded
#         while state["eval_idx"] < target_evals:
#             seed = base_seed + run_id * 10007  # deterministic different seeds
#
#             before_eval = state["eval_idx"]
#             hc_ret = hill_climb_fn(
#                 env_id, base_cfg, param_spec, policy, defaults,
#                 neighbors_per_iter=neighbors_per_iter,
#                 iterations=iterations,
#                 seed=seed
#             )
#             after_eval = state["eval_idx"]
#
#             # Determine if this run found a crash.
#             # Your hill_climb return seems to include 'crashed' and maybe 'best_cfg' etc.
#             run_crashed = bool(hc_ret.get("crashed", False))
#             if run_crashed:
#                 crashes_found_runs += 1
#
#             runs.append({
#                 "run_id": run_id,
#                 "seed": seed,
#                 "evals_consumed": after_eval - before_eval,
#                 "crashed": run_crashed,
#                 "hill_climb_return": hc_ret,
#             })
#
#             run_id += 1
#
#             if after_eval == before_eval:
#                 break
#
#     finally:
#         hill_climbing_module.run_episode = orig_run_episode
#         if orig_record_video is not None:
#             hill_climbing_module.record_video_episode = orig_record_video
#
#     t1 = time.time()
#
#     # ---- matched-budget summary ----
#     total_runs = len(runs)
#     success_rate_per_run = (crashes_found_runs / total_runs) if total_runs > 0 else 0.0
#
#
#     summary = {
#         "method": "hc",
#         "budget_matching": {
#             "target_evals": target_evals,
#             "actual_evals": state["eval_idx"],
#             "neighbors_per_iter": neighbors_per_iter,
#             "iterations": iterations,
#             "base_seed": base_seed,
#         },
#         "efficiency": {
#             "runtime_s": t1 - t0,
#             "total_evaluations": state["eval_idx"],
#         },
#         "success": {
#             "runs": total_runs,
#             "runs_with_collision": crashes_found_runs,
#             "success_rate_per_run": success_rate_per_run,
#         },
#         "failure_discovery": {
#             "found_collision_any": bool(state["best"]["crashed"]) if state.get("best") else False,
#             "distinct_failing_scenarios": len(state.get("distinct_failing", set())),
#             "best_min_distance": (
#                 state["best"]["objectives"].get("min_euclidean_distance", None)
#                 if state.get("best") else None
#             ),
#         },
#         "runs_detail": runs,
#     }
#
#     with open(os.path.join(out_dir, "hc_budget_summary.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)
#
#     return summary

def eval_random_search(random_search_instance: RandomSearch, n_scenarios=50, n_eval=1, seed=42,
                       out_path="eval_outputs/random_eval_summary.json"):
    t0 = time.time()
    random_crash_log = random_search_instance.run_search(n_scenarios, n_eval, seed)
    t1 = time.time()
    total_time = t1 - t0
    iteration_crashes = []
    min_distances_no_crash = []
    crash_configs = []
    fitness_no_crash = []
    print("Started Analysing logs")
    for log in random_crash_log:
        ts = log["time_frames"]
        objectives = compute_objectives_from_time_series(ts)
        min_distance = objectives["min_euclidean_distance"]
        fitness = compute_fitness(objectives)
        config = log["cfg"]
        if log["crashed"]:
            iteration_crashes.append(log["iteration"])
            crash_configs.append(config)
        else:
            min_distances_no_crash.append(min_distance)
            fitness_no_crash.append(fitness)

    success_rate = f"{len(iteration_crashes)} / {n_scenarios}"
    found_collision = len(iteration_crashes) > 0

    first_crash_iteration = min(iteration_crashes).pop() if found_collision else None

    min_distance_no_crash = (
        min(min_distances_no_crash) if min_distances_no_crash else None
    )

    # ---- most problematic config (per-parameter mode) ----
    most_problematic_config = None
    if crash_configs:
        value_counters = defaultdict(Counter)
        for cfg in crash_configs:
            for k, v in cfg.items():
                value_counters[k][v] += 1

        most_problematic_config = {
            k: counter.most_common(1)[0][0]
            for k, counter in value_counters.items()
        }

    summary = {
        "method": "random_search",
        "seed": seed,
        "n_scenarios": n_scenarios,
        "n_eval": n_eval,

        "failure_discovery": {
            "found_collision": found_collision,
            "num_crashes": len(iteration_crashes),
            "success_rate": success_rate,
            "first_crash_iteration": first_crash_iteration + 1,
            "min_distance_if_no_collision": (
                min_distance_no_crash if not found_collision else None
            ),
        },

        "scenario_characteristics": {
            "num_distinct_crash_configs": len(crash_configs),
            "most_problematic_config": most_problematic_config,
        },

        "efficiency": {
            "runtime_s": total_time,
            "total_evaluations": n_scenarios,
            "iterations_to_first_crash": (
                first_crash_iteration + 1 if first_crash_iteration is not None else None
            ),
        },

        "diagnostics": {
            "avg_fitness_no_crash": (
                mean(fitness_no_crash) if fitness_no_crash else None
            ),
            "avg_min_distance_no_crash": (
                mean(min_distances_no_crash) if min_distances_no_crash else None
            ),
        },
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def eval_hill_climbing_early_stop(env_id, base_cfg, param_spec, policy, defaults, max_scenarios, neighbors_per_iter=2,
                                  iterations=5, out_dir="eval_outputs", out_file="hc_summary.json"):
    iteration_crashes = []
    min_distances_no_crash = []
    crash_configs = []
    fitness_no_crash = []
    total_runs = 0
    total_time = 0
    first_crash_iteration: int | None = None

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    for i in range(int(np.ceil(max_scenarios / neighbors_per_iter))):
        if total_runs >= max_scenarios:
            break
        t0 = time.time()
        crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=neighbors_per_iter,
                             iterations=iterations, seed=i * 13 + 4)
        t1 = time.time()
        total_time += t1 - t0
        eval_log = crashes["eval_log"]
        total_runs += len(crashes["eval_log"])
        for entry in eval_log:
            crash = entry["crashed"]

            config = entry["cfg"]
            if crash:
                iteration = (entry["iteration"] + 1) * (entry["neighbor"] + 1)
                iteration_crashes.append(iteration)
                if first_crash_iteration is None:
                    first_crash_iteration = iteration
                crash_configs.append(config)
            else:
                min_distances_no_crash.append(entry["obj"]["min_euclidean_distance"])
                fitness_no_crash.append(entry["fitness"])

    success_rate = f"{len(iteration_crashes)} / {total_runs}"
    found_collision = len(iteration_crashes) > 0

    min_distance_no_crash = (
        min(min_distances_no_crash) if min_distances_no_crash else None
    )

    most_problematic_config = None
    if crash_configs:
        value_counters = defaultdict(Counter)
        for cfg in crash_configs:
            for k, v in cfg.items():
                value_counters[k][v] += 1

        most_problematic_config = {
            k: counter.most_common(1)[0][0]
            for k, counter in value_counters.items()
        }

    summary = {
        "method": "hc",
        "n_scenarios": total_runs,
        "failure_discovery": {
            "found_collision": found_collision,
            "num_crashes": len(iteration_crashes),
            "success_rate": success_rate,
            "first_crash_iteration": first_crash_iteration + 1,
            "min_distance_if_no_collision": (
                min_distance_no_crash if not found_collision else None
            ),
        },

        "scenario_characteristics": {
            "num_distinct_crash_configs": len(crash_configs),
            "most_problematic_config": most_problematic_config,
        },

        "efficiency": {
            "runtime_s": total_time,
            "total_evaluations": total_runs,
            "iterations_to_first_crash": (
                first_crash_iteration + 1 if first_crash_iteration is not None else None
            ),
        },

        "diagnostics": {
            "avg_fitness_no_crash": (
                mean(fitness_no_crash) if fitness_no_crash else None
            ),
            "avg_min_distance_no_crash": (
                mean(min_distances_no_crash) if min_distances_no_crash else None
            ),
        },
    }

    # output
    if not os.path.isdir(os.path.join(os.path.dirname(cur_dir), out_dir)):
        os.mkdir(os.path.join(os.path.dirname(cur_dir), out_dir))

    with open(os.path.join(os.path.dirname(cur_dir), out_dir, out_file), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
