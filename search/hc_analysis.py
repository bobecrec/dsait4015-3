import json
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt


def cfg_to_key(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


def summarize_single_run(eval_log: list[dict], runtime_s: float, seed: int) -> dict:
    cfg_keys = [cfg_to_key(e["cfg"]) for e in eval_log]
    unique_cfgs = set(cfg_keys)

    crash_indices = [i + 1 for i, e in enumerate(eval_log) if e.get("crashed", False)]
    crash_cfgs = [e["cfg"] for e in eval_log if e.get("crashed", False)]

    first_crash_eval = crash_indices[0] if crash_indices else None
    n_crashes = len(crash_indices)

    non_crash = [e for e in eval_log if not e.get("crashed", False)]
    min_dist_no_crash = None
    min_fit_no_crash = None
    if non_crash:
        dists = []
        fits = []
        for e in non_crash:
            if "fitness" in e:
                fits.append(float(e["fitness"]))
            obj = e.get("obj", {})
            if isinstance(obj, dict) and "min_euclidean_distance" in obj:
                dists.append(float(obj["min_euclidean_distance"]))
        min_dist_no_crash = min(dists) if dists else None
        min_fit_no_crash = min(fits) if fits else None

    return {
        "seed": seed,
        "runtime_s": runtime_s,
        "n_evals": len(eval_log),
        "n_unique_cfgs": len(unique_cfgs),
        "unique_ratio": len(unique_cfgs) / max(1, len(eval_log)),
        "n_crashes": n_crashes,
        "first_crash_eval": first_crash_eval,
        "min_distance_no_crash": min_dist_no_crash,
        "min_fitness_no_crash": min_fit_no_crash,
        "crash_cfgs": crash_cfgs,
    }


def aggregate_runs(run_summaries: list[dict]) -> dict:
    n_runs = len(run_summaries)
    success_runs = [r for r in run_summaries if r["n_crashes"] > 0]
    success_rate = len(success_runs) / max(1, n_runs)

    first_crashes = [
        r["first_crash_eval"]
        for r in success_runs
        if r["first_crash_eval"] is not None
    ]
    first_crash_stats = {
        "count": len(first_crashes),
        "mean": float(np.mean(first_crashes)) if first_crashes else None,
        "median": float(np.median(first_crashes)) if first_crashes else None,
        "min": int(np.min(first_crashes)) if first_crashes else None,
        "max": int(np.max(first_crashes)) if first_crashes else None,
    }

    no_crash_distances = [
        r["min_distance_no_crash"]
        for r in run_summaries
        if r["min_distance_no_crash"] is not None and np.isfinite(r["min_distance_no_crash"])
    ]

    no_crash_fitnesses = [
        r["min_fitness_no_crash"]
        for r in run_summaries
        if r["min_fitness_no_crash"] is not None and np.isfinite(r["min_fitness_no_crash"])
    ]

    min_no_crash_distance_overall = (
        float(np.min(no_crash_distances)) if no_crash_distances else None
    )
    min_no_crash_fitness_overall = (
        float(np.min(no_crash_fitnesses)) if no_crash_fitnesses else None
    )

    counters = defaultdict(Counter)
    all_crash_cfgs = []
    for r in run_summaries:
        for cfg in r["crash_cfgs"]:
            all_crash_cfgs.append(cfg)
            for k, v in cfg.items():
                counters[k][v] += 1

    most_common_crash_config_mode = (
        {k: c.most_common(1)[0][0] for k, c in counters.items()}
        if counters else {}
    )

    return {
        "n_runs": n_runs,
        "success_rate": success_rate,
        "first_crash_eval_stats": first_crash_stats,
        "avg_runtime_s": float(np.mean([r["runtime_s"] for r in run_summaries])),
        "avg_unique_ratio": float(np.mean([r["unique_ratio"] for r in run_summaries])),
        "avg_crashes_per_run": float(np.mean([r["n_crashes"] for r in run_summaries])),

        "min_no_crash_distance_overall": min_no_crash_distance_overall,
        "min_no_crash_fitness_overall": min_no_crash_fitness_overall,

        "most_common_crash_config_mode": most_common_crash_config_mode,
    }


def plot_hc_only(run_summaries: list[dict], out_dir: Path):
    out_dir.mkdir(exist_ok=True)

    first_crashes = [r["first_crash_eval"] for r in run_summaries if r["first_crash_eval"] is not None]
    if first_crashes:
        plt.figure(figsize=(6, 4))
        plt.hist(first_crashes, bins=min(10, len(set(first_crashes))))
        plt.title("HC: Distribution of time-to-first-crash (eval index)")
        plt.xlabel("First crash evaluation")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "hc_first_crash_hist.png", dpi=250)
        plt.close()

    plt.figure(figsize=(6, 4))
    x = [r["runtime_s"] for r in run_summaries]
    y = [r["n_crashes"] for r in run_summaries]
    plt.scatter(x, y)
    plt.title("HC: Runtime vs #crashes per run")
    plt.xlabel("Runtime (s)")
    plt.ylabel("# crashes")
    plt.tight_layout()
    plt.savefig(out_dir / "hc_runtime_vs_crashes.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot([r["unique_ratio"] for r in run_summaries], marker="o")
    plt.title("HC: Unique-config ratio per run")
    plt.xlabel("Run index")
    plt.ylabel("Unique configs / evaluations")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "hc_unique_ratio.png", dpi=250)
    plt.close()


def run_hc_multi_analysis(
        hill_climb_fn,
        env_id: str,
        base_cfg: dict,
        param_spec: dict,
        policy,
        defaults: dict,
        seeds: list[int],
        neighbors_per_iter: int = 2,
        iterations: int = 50,
        out_dir: str = "hc_multi_analysis",
        reproducibility_repeats: int = 2,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    run_summaries = []

    # --- Main multi-run loop
    for seed in seeds:
        t0 = time.time()
        out = hill_climb_fn(env_id, base_cfg, param_spec, policy, defaults,
                            neighbors_per_iter=neighbors_per_iter, iterations=iterations, seed=seed)
        t1 = time.time()
        eval_log = out["eval_log"]

        run_summaries.append(summarize_single_run(eval_log, t1 - t0, seed))

    agg = aggregate_runs(run_summaries)
    agg["settings"] = {
        "neighbors_per_iter": neighbors_per_iter,
        "iterations": iterations,
        "n_runs": len(seeds),
        "seeds": seeds,
    }

    # Save outputs
    with open(out_dir / "hc_runs.json", "w") as f:
        json.dump(run_summaries, f, indent=2)

    with open(out_dir / "hc_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    plot_hc_only(run_summaries, out_dir / "plots")

    return agg
