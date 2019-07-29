import gym
import air_sim_deep_drone
import time
import numpy as np


def list_all_envs():
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        print(env_type, " --> ", env.id)


def find_env_in_gym(name):
    print("===== try to find %s in gym envs =====" % name)
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        if (env._entry_point.find(name) >= 0):
            print(env._entry_point)
            print(env_type, " --> ", env.id)
            print("===== Finded =====")
            return

    print("===== Cannot find envs =====")


def display_gym_env():
    env = gym.make("Humanoid-v2")
    print(env.action_space)
    print(env.observation_space)
    print(env.reset())


if __name__ == "__main__":
    # list_all_envs()
    # display_gym_env()
    # find_env_in_gym("deep_drone")
    # env = gym.make('Deepdrone-v0')
    env = gym.make('Crossgap-v0')
    env.print_help()
    print(env.action_space)
    print(env.observation_space)
    ac = env.action_space.sample()
    print(env.reset())
    env.if_log = 1
    for i in range(1000):
        # print(i)
        # obs, reward, done, _ =  env.step(env.action_space.sample())
        obs, reward, done, _ =  env.step(env.action_space.sample())
        # print(obs)
        if(done):
            pass
            # env.reset()
        time.sleep(0.001)
    print(env.reset())
    # env.plot_log()
    # print(env_deep_drone)
