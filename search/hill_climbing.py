"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from envs.highway_env_utils import run_episode, record_video_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================
def time_to_crash(x, y, vx, vy, v, carx, cary, w, l):
    # we return 100 instead of infinity to avoid numerical errors
    if v < 1e-6:
        return 100.0
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return 100.0

    # Normalize velocity direction
    vx_norm = vx / v
    vy_norm = vy / v

    # Vector from ego to car
    dx = carx - x
    dy = cary - y

    # Check if moving towards the car (dot product > 0)
    dot = dx * vx_norm + dy * vy_norm
    if dot <= 0:
        return 100.0 # will not crash, just return big number

    # Project velocity onto the vector to the car
    distance = (dx ** 2 + dy ** 2) ** 0.5
    collision_distance = ((w / 2) ** 2 + (l / 2) ** 2) ** 0.5  # bounding box is a circle

    if distance < collision_distance:
        return 0.0  # Already overlapping

    # Time = distance / speed (projected)
    return (distance - collision_distance) / v


def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    # TODO (students)
    crashed = 0
    min_dist = float("inf")
    avg_min_dist = 0
    max_neighboring_cars_ahead = 0
    avg_neighbouring_cars_ahead = 0
    min_crash_time = float('inf')
    avg_crash_time = 0

    for frame in time_series:
        if frame.get("crashed", False):
            crashed = 1

        ego = frame.get("ego", None)
        if ego is None:
            continue
        pos_ego = ego.get("pos", None)
        ego_lane = ego.get("lane_id", None)
        if pos_ego is None or ego_lane is None:
            continue

        heading = ego.get("heading", 0)
        vy = np.sin(heading)
        vx = np.cos(heading)

        others = frame.get("others", [])
        if not others:
            continue

        min_crash_this_frame = float('inf')
        # Count cars in neighboring lanes that are ahead of ego
        neighboring_cars_ahead = 0
        for o in others:
            pos_o = o.get("pos", None)
            lane_o = o.get("lane_id", None)
            if pos_o is None or lane_o is None:
                continue

            # Check if vehicle is in a neighboring lane (adjacent lane)
            is_neighboring_lane = abs(lane_o - ego_lane) == 1

            # Check if vehicle is ahead (x position greater than ego)
            is_ahead = pos_o[0] > pos_ego[0]

            if is_neighboring_lane and is_ahead:
                neighboring_cars_ahead += 1

            # Also compute minimum distance
            dx = pos_o[0] - pos_ego[0]
            dy = pos_o[1] - pos_ego[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < min_dist:
                min_dist = d

            crash = time_to_crash(pos_ego[0], pos_ego[1], vx, vy, ego.get("speed"), pos_o[0], pos_o[1], o.get("width"),
                                  o.get("length"))
            if crash < min_crash_this_frame:
                min_crash_this_frame = crash

        avg_crash_time += min_crash_this_frame
        if min_crash_this_frame < min_crash_time:
            min_crash_time = min_crash_this_frame

        avg_neighbouring_cars_ahead += neighboring_cars_ahead
        # Track the maximum number of neighboring cars ahead across all frames
        if neighboring_cars_ahead > max_neighboring_cars_ahead:
            max_neighboring_cars_ahead = neighboring_cars_ahead

    return {
        "crashed": crashed,
        "min_euclidean_distance": min_dist,
        "max_neighboring_cars_ahead": max_neighboring_cars_ahead,
        "avg_neighboring_cars_ahead": float(avg_neighbouring_cars_ahead) / len(time_series),
        "min_crash_time": min_crash_time,
        "avg_crash_time": avg_crash_time / len(time_series)
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    # TODO (students)
    print(objectives)
    if objectives["crashed"] == 1:
        fitness = - 1.0
    else:
        fitness = 10 * objectives['min_crash_time'] + objectives['min_euclidean_distance']
    return fitness


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
        cfg: Dict[str, Any],
        param_spec: Dict[str, Any],
        rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    # single parameter mutation
    mod_cfg = copy.deepcopy(cfg)
    keys = ["vehicles_count", "lanes_count", "initial_spacing", "ego_spacing", "initial_lane_id"]

    k = rng.choice(keys)
    s = param_spec[k]
    
    # gaussian noise centered on current value, with std proportional to range
    current_val = mod_cfg.get(k, (s["min"] + s["max"]) / 2)
    param_range = s["max"] - s["min"]
    std = param_range * 0.4
    
    # Sample with Gaussian noise and clamp to bounds
    new_v = rng.normal(current_val, std)
    new_v = np.clip(new_v, s["min"], s["max"])
    
    if s["type"] == "int":
        new_v = int(round(new_v))
        # retry if same value
        if new_v == mod_cfg[k]:
            new_v = int(round(np.clip(rng.normal(current_val, std), s["min"], s["max"])))
    else:
        new_v = float(new_v)

    mod_cfg[k] = new_v

    if k == "lanes_count":
        mod_cfg["initial_lane_id"] = int(np.clip(mod_cfg.get("initial_lane_id", 0), 0, mod_cfg["lanes_count"] - 1))

    if k == "initial_lane_id":
        lanes = int(mod_cfg.get("lanes_count", 3))
        mod_cfg["initial_lane_id"] = int(np.clip(mod_cfg["initial_lane_id"], 0, lanes - 1))

    return mod_cfg
    # TODO (students)
    # raise NotImplementedError


def sample_random_config(base_cfg, param_spec, rng):
    cfg = {}
    lanes = None
    if "lanes_count" in param_spec:
        s = param_spec["lanes_count"]
        lanes = int(rng.integers(s["min"], s["max"] + 1))
        cfg["lanes_count"] = lanes

    for k, s in param_spec.items():
        if k == "lanes_count":
            continue
        if k == "initial_lane_id":
            lanes = lanes or 3
            cfg[k] = int(rng.integers(0, lanes))
            continue
        if s["type"] == "int":
            cfg[k] = int(rng.integers(s["min"], s["max"] + 1))
        elif s["type"] == "float":
            cfg[k] = float(rng.uniform(s["min"], s["max"]))

    for k, v in base_cfg.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
        env_id: str,
        base_cfg: Dict[str, Any],
        param_spec: Dict[str, Any],
        policy,
        defaults: Dict[str, Any],
        seed: int = 0,
        iterations: int = 100,
        neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    # TODO (students): choose initialization (base_cfg or random scenario)
    current_cfg = sample_random_config(base_cfg, param_spec, rng)

    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    history = [best_fit]

    log_for_eval = []
    stalled = 0
    # TODO (students): implement HC loop
    crash = False
    for i in tqdm(range(iterations), desc="Running Hill Climbing Iterations"):
        old_fit = best_fit
        if stalled >= 3:
            # Restart from a new random configuration - reset everything
            current_cfg = sample_random_config(base_cfg, param_spec, rng)
            restart_seed = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, current_cfg, policy, defaults, restart_seed)
            obj = compute_objectives_from_time_series(ts)
            cur_fit = compute_fitness(obj)
            # Reset best to the new random config
            best_fit = cur_fit
            best_cfg = copy.deepcopy(current_cfg)
            best_obj = dict(obj)
            best_seed_base = restart_seed
            old_fit = best_fit
            if crashed:
                best_fit = -1000
                crash = True
                print("CAR CRASHED on restart!!!")
                break
            stalled = 0
            print(f"Restarted at iteration {i} with fitness {cur_fit}")

        for j in range(neighbors_per_iter):
            neighbor_seed = int(rng.integers(1e9))
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)
            crashed, ts = run_episode(env_id, neighbor_cfg, policy, defaults, neighbor_seed)
            objectives = compute_objectives_from_time_series(ts)
            new_fit = compute_fitness(objectives)
            crash |= crashed
            log_for_eval.append(
                {"iteration": i, "neighbor": j, "crashed": crashed, "obj": objectives, "fitness": new_fit,
                 "cfg": copy.deepcopy(neighbor_cfg), "seed": neighbor_seed})
            if new_fit < best_fit or crashed:
                best_fit = new_fit
                best_cfg = neighbor_cfg
                current_cfg = neighbor_cfg
                best_obj = dict(objectives)
                best_seed_base = neighbor_seed
                if crash:
                    best_fit = -1000
                    print("CAR CRASHED!!!")
                    break
        if old_fit == best_fit:
            stalled += 1

        history.append(best_fit)
        print(f"Iteration {i}: {best_fit}")

        if crash:
            print(f"💥 Collision: scenario {i}")
            record_video_episode(env_id, best_cfg, policy, defaults, best_seed_base, out_dir="videos")
            break
    # print(best_cfg)
    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "history": history,
        "crashed": best_fit < 0,
        "eval_log": log_for_eval
    }

    # - generate neighbors
    # - evaluate
    # - pick best
    # - accept if improved
    # - early stop on crash (optional)
