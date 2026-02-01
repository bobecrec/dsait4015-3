from __future__ import annotations

import os
import json
import time
from collections import defaultdict, Counter

from numpy import mean

from search.random_search import RandomSearch
from search.hill_climbing import compute_objectives_from_time_series, compute_fitness, hill_climb


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

    first_crash_iteration = min(iteration_crashes) if found_collision else None

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
            "first_crash_iteration": first_crash_iteration + 1 if first_crash_iteration is not None else None,
            "min_distance_if_no_collision": (
                min_distance_no_crash if not found_collision else None
            ),
        },

        "scenario_characteristics": {
            "num_distinct_crash_configs": len(crash_configs),
            "most_problematic_config": most_problematic_config,
            "configs": crash_configs,
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
            "min_distance_no_crash": min(min_distances_no_crash),
            "min_fitness_no_crash": min(fitness_no_crash),
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
    # total_runs = 0
    total_time = 0
    first_crash_iteration: int | None = None

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    seed_i = 0
    total_runs_per_iteration = []
    true_runs = 0
    while true_runs < max_scenarios:
        t0 = time.time()
        crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=neighbors_per_iter,
                             iterations=iterations, seed=seed_i * 15 + 4)
        t1 = time.time()
        total_time += t1 - t0
        eval_log = crashes["eval_log"]
        # total_runs += len(crashes["eval_log"])
        # print(f"This are the total runs so far you mongol!!!!!!!!!!!!!!!!!!!!!!!!!! - {total_runs}")
        seed_i += 1
        total_runs_per_iteration.append(len(eval_log))
        remaining = max_scenarios - true_runs
        take = min(len(eval_log), remaining)
        for entry in eval_log[:take]:
            # print(f"Eval Log entry in eval: {entry}")
            # if total_runs > max_scenarios and i >= total_runs-max_scenarios:
            #     print("You are in the fucking break statement, moron ?????????????!!!!!!!!!!!!!!!!!!")
            #     break
            true_runs += 1
            crash = entry["crashed"]

            config = entry["cfg"]
            if crash:

                global_eval_idx = true_runs
                iteration_crashes.append(global_eval_idx)

                if first_crash_iteration is None:
                    first_crash_iteration = global_eval_idx
                crash_configs.append(config)
            else:
                min_distances_no_crash.append(entry["obj"]["min_euclidean_distance"])
                fitness_no_crash.append(entry["fitness"])
        print(f"This are the true runs so far you mongol!!!!!!!!!!!!!!!!!!!!!!!!!! - {true_runs}")


    success_rate = f"{len(iteration_crashes)} / {true_runs}"
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
        "n_scenarios": true_runs,
        "failure_discovery": {
            "found_collision": found_collision,
            "num_crashes": len(iteration_crashes),
            "success_rate": success_rate,
            "first_crash_iteration": first_crash_iteration,
            "min_distance_if_no_collision": (
                min_distance_no_crash if not found_collision else None
            ),
        },

        "scenario_characteristics": {
            "num_distinct_crash_configs": len(crash_configs),
            "most_problematic_config": most_problematic_config,
            "crash_configs": crash_configs,
        },

        "efficiency": {
            "runtime_s": total_time,
            "total_evaluations": true_runs,
            "iterations_to_first_crash": (
                first_crash_iteration if first_crash_iteration is not None else None
            ),
            "times_to_find_crash": total_runs_per_iteration,
        },

        "diagnostics": {
            "avg_fitness_no_crash": (
                mean(fitness_no_crash) if fitness_no_crash else None
            ),
            "avg_min_distance_no_crash": (
                mean(min_distances_no_crash) if min_distances_no_crash else None
            ),
            "min_distance_no_crash":
                min(min_distances_no_crash),
            "min_fitness_no_crash": min(fitness_no_crash),
        },
    }

    # output
    if not os.path.isdir(os.path.join(os.path.dirname(cur_dir), out_dir)):
        os.mkdir(os.path.join(os.path.dirname(cur_dir), out_dir))

    with open(os.path.join(os.path.dirname(cur_dir), out_dir, out_file), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
