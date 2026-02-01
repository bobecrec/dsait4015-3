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
    #     if crashes['crashed']:
    #         crashed += 1
    #
    # print("CAR CRASHED ", crashed)
    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=50, seed=11)

    # print(f"✅ Found {len(c
    # rashes)} crashes.")
    # if crashes:
    #    print(crashes)
    # eval_random_search(random_search_instance=search, n_scenarios=50,n_eval=1, seed=11)
    # eval_hill_climbing_early_stop(env_id, base_cfg, param_spec, policy, defaults, 50, neighbors_per_iter=5, iterations=10)



if __name__ == "__main__":
    main()
