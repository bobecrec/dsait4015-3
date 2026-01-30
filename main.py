from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import hill_climb
import search.hill_climbing as hc_impl


from search.eval_searches import eval_random_search, eval_hill_climbing_early_stop


def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # Standard Runs of Searches

    # crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=5, iterations=10,
    #                      seed=9653893457)
    # crashed = 0
    # for i in range(10):
    #     crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=2, iterations=5, seed=i*13+4)
    #     print(crashes)
    #     if crashes['crashed']:
    #         crashed += 1

    # print("CAR CRASHED ", crashed)
    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=50, seed=11)

    # print(f"✅ Found {len(c
    # rashes)} crashes.")
    # if crashes:
    #    print(crashes)

    # Evaluation Runs
    # rand = eval_random_search(
    #     env_id, base_cfg, param_spec, policy, defaults,
    #     n_scenarios=50, n_eval=1, seed=11,
    #     out_dir="eval_outputs/random_seed11",
    #     save_videos=False
    # )
    # print("Random summary written:", rand["paths"]["summary"])
    #
    # hc = eval_hill_climb(
    #     env_id, base_cfg, param_spec, policy, defaults,
    #     neighbors_per_iter=5, iterations=10, seed=9653893457,
    #     out_dir="eval_outputs/hc_seed9653893457",
    #     save_videos=False
    # )
    # print("HC summary written:", hc["paths"]["summary"])
    # target_evals = 50 * 1
    # hc_budget_summary = eval_hc_until_budget(
    #     env_id=env_id,
    #     base_cfg=base_cfg,
    #     param_spec=param_spec,
    #     policy=policy,
    #     defaults=defaults,
    #     hill_climb_fn=hill_climb,
    #     hill_climbing_module=hc_impl,
    #     target_evals=target_evals,
    #     neighbors_per_iter=5,
    #     iterations=10,
    #     base_seed=9653893457,
    #     out_dir="eval_outputs/hc_budget"
    # )
    #
    # print("HC budget-matched done")
    # print("HC success rate:",
    #       hc_budget_summary["success"]["success_rate_per_run"])
    # eval_random_search(random_search_instance=search, n_scenarios=50,n_eval=1, seed=11)
    eval_hill_climbing_early_stop(env_id, base_cfg, param_spec, policy, defaults, 2, neighbors_per_iter=2, iterations=10)



if __name__ == "__main__":
    main()
