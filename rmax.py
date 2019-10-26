import gym
import numpy as np
import random

class MDP:
    def __init__(self, env, discount=1.0, iterThresh=0.00001):
        self.U = {}
        self.env = env
        self.gamma = discount
        self.theta = iterThresh
        self.na = env.action_space.n
        self.ns = env.observation_space.n
        self.policy = np.ones([self.ns, self.na]) / self.na
        #print('made blank policy: ', self.policy)
        
    def getAction(self, s):
        action_val = max(self.policy[s])
        action_idx = [i for i, j in enumerate(self.policy[s]) if j == action_val]
        if len(action_idx) > 1:
            action_idx = action_idx[ round(random.random() * (len(action_idx)-1)) ]
        return action_idx
        
    def evaluate(self):
        V = np.zeros(self.ns)
        while True:
            delta = 0
            for s in range(self.ns):
                v = 0
                for a, a_prob in enumerate(self.policy[s]):
                    for prob, next_state, reward, done in env.P[s][a]:
                        v += a_prob * prob * (reward + self.gamma * V[next_state])
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
                
            if delta < self.theta:
                break
        return np.array(V)

    def improve(self):
        # evaluate current policy
        v = self.evaluate()

        #print('evaluated, state utils are: ', v)

        for s in range(self.ns):
            row = np.zeros(self.na)
            for a, a_prob in enumerate(self.policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    row[a] += a_prob * prob * v[next_state]

            # update policy with new action likelyhoods
            self.policy[s] = row


# make two identical environments, one for planning and one
# for executing actions and observing reward info to propogate
# back into the planning mdp

#planEnv = gym.make('FrozenLake8x8-v0')
#realEnv = gym.make('FrozenLake8x8-v0')
planEnv = gym.make('FrozenLake-v0')
realEnv = gym.make('FrozenLake-v0')

# make MDPs for each of the two environments
planMDP = MDP(planEnv)
realMDP = MDP(realEnv)


done = False

while not done:
    env.render()
    print('in state ', nextState)
    
    action = knownPolicy.getAction(nextState)
    print('  taking action ', action)
    
    nextState, reward, done, info = env.step(env.action_space.sample())

    knownPolicy.improve()

    print('new policy: ', knownPolicy.policy)
env.render()
