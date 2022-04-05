from mjrl.utils.gym_env import GymEnv
import mj_allegro_envs
import numpy as np
def main():
    env_name = 'block-v3'
#    env_name = 'bottlecap-v0'
    e = GymEnv(env_name)
    e.set_seed(123)
    e.env.reset()
    for i in range(100000):
        # e.get_end_effector_coordinates()
        action = e.env.env.action_space.sample()
        # import pdb
        # pdb.set_trace()
        # action = np.ones(16) *3.6
        action = np.zeros(16)
        action[8] = 1
        # action[-4:]= [0.77840335, 2.4313333, 2.40583211, 1.37569983]
        print(action)
        obs, reward, done, info = e.env.step(action)
        e.render()
        if(i%600 ==0):
            print("RESETTING")
            e.env.env.reset()

if __name__ == '__main__':
    main()

