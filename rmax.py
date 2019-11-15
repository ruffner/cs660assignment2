import gym
import numpy as np
import random
import time
import rl1

env       = gym.make('FrozenLake-v0')
theta     = 0.001
gamma     = 0.8
tk_thresh = 5
testruns = 100 # number of times to run policy to determin avg reward
episodes = 10

np.random.seed(np.int(time.time()))

class Policy:
    def __init__(self):
        self.policy = np.random.randint(0,
                                        env.action_space.n,
                                        size=[env.observation_space.n])
        #self.policy = np.zeros([env.observation_space.n, env.action_space.n])
        self.values = np.zeros(env.observation_space.n)
        self.Rsa    = np.zeros([env.observation_space.n, env.action_space.n])
        self.Nsas   = np.zeros([env.observation_space.n,
                                env.action_space.n,
                                env.observation_space.n])
        self.known_arcs = 0
        
    def getRs(self):
        return self.Rs

    def getNsas(self):
        return self.Nsas
    
    def getNsa(self):
        return np.sum(self.Nsas, axis=2)

    def getK(self):
        return np.argwhere(self.getNsa() > tk_thresh)

    def getPolicy(self):
        return self.policy

    def getValues(self):
        return self.values
    
    def step(self, s, a, r, sp):
        self.Nsas[s, a, sp] += 1
        self.Rsa[s][a] += r

    def selectAction(self, s):
        return self.policy[s]
    
        #actions = self.policy[s]
        #if np.amax(actions).size > 1:
        #    return np.random.randint(0, np.amax(actions).size)
        #else:
        #    return np.argmax(actions)

    def should_replan(self):
        K = self.getK()
        if K.shape[0] > self.known_arcs:
            self.known_arcs = K.shape[0]
            return True
        else:
            return False

    def getMDP(self):
        P = np.zeros(self.Nsas.shape)
        K = self.getK()
        R = np.ones(self.getNsa().shape)

        for s in range(P.shape[0]):
            for a in range(P.shape[1]):
                for sp in range(P.shape[2]):
                    for t in K:
                        if t[0] == s and t[1] == a:
                            P[s][a][sp] = self.Nsas[s][a][sp] / self.getNsa()[s][a]
                        else:
                            if s == sp:
                                P[s][a][sp] = 1
                            else:
                                P[s][a][sp] = 0
        for s in range(R.shape[0]):
            for a in range(R.shape[1]):
                for t in K:
                    if not (t[0] == s and t[1] == a):
                        R[s][a] = 1
        return P, R

'''
    def iterate(self):
        V = self.values
        P, R = self.getMDP()


        # ---------------------------------    evaluation
        while True:
            delta = 0
            
            # sum over s
            for s in range(self.policy.shape[0]):
                v = V[s]

                # sum over s'
                for sp in range(self.policy.shape[0]):
                    a = self.selectAction(s)
                    v += P[s, self.selectAction(s), sp] * ( R[s][a] + gamma * V[sp])

                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
                
            if delta < theta:
                break

        print('evaluated, state utils are: ', v)

        
        # --------------------------------   improvement
        
        stable = True
        
        for s in range(self.policy.shape[0]):
            row = np.zeros(env.action_space.n)
            for a, a_prob in enumerate(self.policy[s]):
                for sp in range(self.policy.shape[0]):
                    row[a] += a_prob * P[s][a][sp] * V[sp]

            # update policy with new action likelyhoods
            self.policy[s] = row
'''


#######################################################
#######################################################

pknown = Policy()

print('initial random policy: ', pknown.policy)

def evalPolicy(Q, trials):
    # evaluate the policy
    total_reward = 0
    for _ in range(trials):
        state = env.reset()
        while(True):
            action = Q[state]
            next_state,reward,done,_ = env.step(action)
            state = next_state
        
            total_reward += reward
    
            if done: break

    total_reward /= trials
    return total_reward

print(evalPolicy(pknown.policy, testruns))

for ep in range(episodes):

    s = env.reset()

    while True:
        # select and action from our known mdp
        a = pknown.selectAction(s)
        
        # execute and observe in the real environment
        sp, r, terminal, _ = env.step(a)

        # update our policy class so we know when we have
        # a new known transition
        pknown.step(s, a, r, sp)

        if terminal or pknown.should_replan():
            # first update env to reflect our known mdp
#            for s in range(env.nS):
 #               maxvsa = -1
  #              for a in range(env.nA):
   #                 env.P[s][a][
            
            #print('K:', pknown.getK())
            
            val_func, iters = rl1.value_iteration(env, gamma, max_iterations=100)
            policy = rl1.value_function_to_policy(env, gamma, val_func)
            pknown.policy = policy
            #print('new policy: ',policy)
            break
        
        s = sp

    #if ep%5==0:
    #print('after episode ',ep)
    print(evalPolicy(pknown.policy, testruns))
