import numpy as np
import gym

# environment
##############################################
# ive just been switching this one manually
env = gym.make('FrozenLake8x8-v0')


###############################################
# parameters
alpha = 0.05
gamma = 0.95
epsilon = 0.9
max_steps = 2000
test_runs = 1000

# greedy action selector
# choose an action, greedily with prob e
# when you are in state s from policy q
def chooseAction(Q, s, e):
    if np.random.uniform(0, 1) < e:
        return np.argmax( Q[s, :] )
    else:
        return env.action_space.sample()

# average reward calc
# same as in qlearn and rmax files
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

    
# sarsa algorithm, add one action/state lookahead to q learning
# print out average reward 
def sarsa(alpha, gamma, epsilon, episodes, max_steps):
    n_states, n_actions = env.observation_space.n, env.action_space.n

    # in testing, random policy initialization yielded better
    # results than all ones
    Q = np.random.random([n_states, n_actions])

    for episode in range(episodes):
        total_reward = 0
        s = env.reset()
        a = chooseAction(Q, s, epsilon)
        t = 0
        done = False

        if episode % 1000 == 0:
            rew = evalPolicy(Q, test_runs)
            print('Average reward after ', episode, ' episodes: ', rew)

        
        while t < max_steps:
            t += 1

            # observe s' and r' from the environment
            sp, reward, done, info = env.step(a)
            total_reward += reward

            # choose a' based on s'
            ap = chooseAction(Q, sp, epsilon)
            
            if done:
                Q[s, a] += alpha * ( reward  - Q[s, a] )
            else:
                Q[s, a] += alpha * ( reward + (gamma * Q[sp, ap] ) - Q[s, a] )

            # update to new states
            s, a = sp, ap
            
            if done:
                break
            
    return Q


sarsa(alpha, gamma, epsilon, 10000, max_steps)
