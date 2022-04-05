from mjrl.utils.gym_env import GymEnv
import mj_allegro_envs
import numpy as np

def main():
    env_name = 'denseblock-v5'
    #[90,0,0] [90,14,0] [90,29,0] [90,45,0]
    goals = [[0.7,0.7,0.0,0.0],[0.701,0.701,0.092,0.092],[0.683,0.683,0.183,0.183],[0.653,0.653,0.271,0.271],[0.5,0.5,0.5,0.5]]


    e = GymEnv(env_name)
    e.env.set_mix_goals(1) #switch goals half the time
    e.set_seed(123)
    e.env.reset()
    goal_idx = 1

    

    s = 0.2
    for i in range(100000):
        action = e.env.env.action_space.sample()
        action = np.ones(action.shape[0]) *10
        # action[-4] = 5
        obs, reward, done, info = e.env.step(action)
        e.render()
        if(i%60 ==0):

            print("RESETTING")
            if(goal_idx< len(goals)):
            	e.env.env.increase_goal_difficulty(goals[goal_idx])

            	print("new goal: " + str(goals[goal_idx]))
            	goal_idx += 1
            # if(i==60): e.env.env.set_random_start(s)
            e.env.env.reset()

if __name__ == '__main__':
    main()




