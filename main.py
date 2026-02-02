from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import hill_climb
import search.hill_climbing as hc_impl
from search.hill_climbing import hill_climb
from search.hc_analysis import run_hc_multi_analysis

from search.eval_searches import eval_random_search, eval_hill_climbing_early_stop


def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    #run hill climb on its own
    # crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=5, iterations=10,
    # seed=9653893457)

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # run random search on its own
    # crashes = search.run_search(n_scenarios=50, seed=1123)

    # analysis of random search performance
    eval_random_search(random_search_instance=search, n_scenarios=50, n_eval=1, seed=121)

    # analysis of hill climb with early stop
    # eval_hill_climbing_early_stop(env_id, base_cfg, param_spec, policy, defaults, 50, neighbors_per_iter=5, iterations=10)

    # analysis of hill climb performance on its own
    # agg = run_hc_multi_analysis(
    #     hill_climb_fn=hill_climb,
    #     env_id=env_id,
    #     base_cfg=base_cfg,
    #     param_spec=param_spec,
    #     policy=policy,
    #     defaults=defaults,
    #     seeds=[4, 17, 30, 43, 56, 69, 82, 95, 108, 121],
    #     neighbors_per_iter=2,
    #     iterations=5,
    #     out_dir="eval_outputs/hc_multi_analysis",
    # )

    # print(agg)



if __name__ == "__main__":
    main()
