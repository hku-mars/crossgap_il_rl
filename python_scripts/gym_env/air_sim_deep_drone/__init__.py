import logging
from gym.envs.registration import register
from air_sim_deep_drone.deep_drone_env import Deep_drone_env
# from air_sim_deep_drone.cross_grap_env import Cross_gap_env_v1
from air_sim_deep_drone.cross_grap_env_v2 import Cross_gap_env

register(
    id='Deepdrone-v0',
    entry_point='air_sim_deep_drone:Deep_drone_env',
)

register(
    id='Crossgap-v0',
    entry_point='air_sim_deep_drone:Cross_gap_env',
)

register(
    id='Crossgap-v2',
    entry_point='air_sim_deep_drone:Cross_gap_env',
)