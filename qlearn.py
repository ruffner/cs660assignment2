import numpy as np
import gym

# set to True to use the 8x8 frozen lake
bigboy = False

# environment parameters
if bigboy:
    # parameters for 8x8 environment
    env     = gym.make('FrozenLake8x8-v0')
    alpha   = 0.05
    epsilon = 0.3
    gamma   = 0.95
else:
    # parameters for 4x4 environment
    env     = gym.make('FrozenLake-v0')
    alpha   = 0.015
    epsilon = 0.3
    gamma   = 0.95

# choose an action, greedily with prob e
# when you are in state s from policy q
def chooseAction(Q, s, e):
    if np.random.random() < epsilon:
        return np.argmax( Q[s, :] )
    else:
        return env.action_space.sample()

# calculate td error
# policy q, reward r, state sprime and discount
# factor gamma
def samplePolicy(Q, r, sp):
    return r + gamma * np.amax( Q[sp, :] )

# preform a td update
def qUpdate(Q, s, a, r, sp):
    return (1 - alpha) * Q[s, a] + alpha * samplePolicy(Q, r, sp)

# episode loop
def qLearn(nepisodes):

    # Q = [S x A]
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # policy optimality will be judged on
    # the number of episodes since
    # the agent last won a game, my thought was, as the
    # policy converges, this number should shrink. 
    windex = 0
    
    for t in range(1, nepisodes+1 ):        

        
        #print('win rate: ', 1 / (t - windex+1))
        
        s = env.reset()
        while True:
            a = chooseAction(Q, s, epsilon)

            # observe s' and r 
            sp, r, done, info = env.step(a)

            # preform q-update
            Q[s,a] = qUpdate(Q, s, a, r, sp)
            
            s = sp

            # if the state was terminal and we reached the goal
            # update the win index index
            if done:
                if r == 1:
                    windex = t
                break
        
    # return the learned policy
    return Q

###################################################################

def evalPolicy(Q, trials):
    # evaluate the policy
    total_reward = 0
    for _ in range(trials):
        state = env.reset()
        while(True):
            action = np.argmax( Q[state, :] )
            next_state,reward,done,_ = env.step(action)
            state = next_state
        
            total_reward += reward
    
            if done: break

    total_reward /= trials
    return total_reward

####################################################################

# search space for parameter optimization
episode_space    = np.linspace(5000, 20000, num=4)

# number of games to play when evaluating a policy
eval_trials = 10

# header output for csv log
print('# episodes,\t average_reward')

for e in range(episode_space.size):
    policy = qLearn(np.int(episode_space[e]))
    reward = evalPolicy(policy, eval_trials)
    print(np.int(episode_space[e]), ',\t\t' , np.around(reward,decimals=2))
