from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import hill_climb

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=2, iterations=5, seed=1)
    crashed = 0
    # for i in range(10):
    #     crashes = hill_climb(env_id, base_cfg, param_spec, policy, defaults, neighbors_per_iter=2, iterations=5, seed=i*13+4)
    #     print(crashes)
    #     if crashes['crashed']:
    #         crashed += 1

    print("CAR CRASHED ", crashed)
    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=50, seed=11)

    # print(f"✅ Found {len(c
    # rashes)} crashes.")
    #if crashes:
    #    print(crashes)

if __name__ == "__main__":
    main()