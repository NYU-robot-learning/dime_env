from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

# Allegro hand manipulating block, rotating 90 deg
register(
	id='block-v3',
	entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:BlockEnvV3',
    max_episode_steps=200,
)
register(
	id='denseblock-v3',
	entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:DenseBlockEnvV3',
    max_episode_steps=200,
)
register(
    id='startblock-v3',
    entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:RandomStartEnvV3',
    max_episode_steps=200,
)
# Allegro hand manipulating block 90 deg with only last finger and thumb
register(
    id='lim-block-v0',
    entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:LimBlockEnvV0',
    max_episode_steps=200,
)
register(
    id='block-v5',
    entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:BlockEnvV5',
    max_episode_steps=200,
)
register(
    id='denseblock-v5',
    entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:DenseBlockEnvV5',
    max_episode_steps=200,
)


register(
    id='allegro-rectangle-v0',
    entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:RectangleEnvV0',
    max_episode_steps=700,
)

register(
    id='bottlecap-v0',
    entry_point='mj_allegro_envs.allegro_hand_manipulation_suite:BottleCapEnvV0',
    max_episode_steps=100,
)


from mj_allegro_envs.allegro_hand_manipulation_suite.block_v3 import BlockEnvV3
from mj_allegro_envs.allegro_hand_manipulation_suite.block_v5 import BlockEnvV5
from mj_allegro_envs.allegro_hand_manipulation_suite.block_v5 import DenseBlockEnvV5
from mj_allegro_envs.allegro_hand_manipulation_suite.block_v3 import DenseBlockEnvV3

from mj_allegro_envs.allegro_hand_manipulation_suite.rectangle_v0 import RectangleEnvV0
from mj_allegro_envs.allegro_hand_manipulation_suite.bottlecap_v0 import BottleCapEnvV0

