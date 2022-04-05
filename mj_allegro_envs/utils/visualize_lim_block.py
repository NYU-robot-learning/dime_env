from mjrl.utils.gym_env import GymEnv
import mj_allegro_envs
import numpy as np
def main():
    env_name = 'lim-block-v0'
    e = GymEnv(env_name)
    e.set_seed(123)
    e.env.reset()
    for i in range(100000):
        action = e.env.action_space.sample()
        # action = np.ones(8)
        print(action)
        obs, reward, done, info = e.env.step(action)
        e.render()
        if(i%600 ==0):
            print("RESETTING")
            e.env.env.reset()

if __name__ == '__main__':
    main()
