# build up a env in gym
# Reference: https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai
from setuptools import setup

setup(name='air_sim_deep_drone',
      version='0.0.3',
      author="ziv.lin",
      install_requires=['gym>=0.2.3',
                        'pandas',
                        'cfg_load',
                        'airsim']
)