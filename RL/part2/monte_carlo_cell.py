import numpy as np
import random

class ExactMCAgent:
    """
    The exact Monte-Carlo agent.
    This agents performs value update as follows:
    V(s) <- s(s) / n(s)
    Q(s,a) <- s(s,a) / n(s,a)
    """

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float):
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon

        self._eps = 1e-10  # for stable computation of V and Q. NOT the one for e-greedy !
        #cellular automata 
        self.cellcount = 20
        self.offendvalue = '0'
        self.x=0
        self.prev=0 
        self.universe = ''.join(random.choice('01') for i in range(self.cellcount))
        # Initialize statistics
        self.n_v = None
        self.s_v = None
        self.n_q = None
        self.s_q = None
        self.reset_statistics()

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

        # Initialize "policy Q"
        # "policy Q" is the one used for policy generation.
        self._policy_q = None
        self.reset_policy()

    def reset_statistics(self):
        self.n_v = np.zeros(shape=self.num_states)
        self.s_v = np.zeros(shape=self.num_states)

        self.n_q = np.zeros(shape=(self.num_states, self.num_actions))
        self.s_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_policy(self):
        self._policy_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset(self):
        self.reset_statistics()
        self.reset_values()
        self.reset_policy()

    def update(self, episode):
        states, actions, rewards = episode

        # reversing the inputs!
        # for efficient computation of returns
        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        iter = zip(states, actions, rewards)
        cum_r = 0
        for s, a, r in iter:
            cum_r *= self.gamma
            cum_r += r

            self.n_v[s] += 1
            self.n_q[s, a] += 1

            self.s_v[s] += cum_r
            self.s_q[s, a] += cum_r

    def compute_values(self):
        self.v = self.s_v / (self.n_v + self._eps)
        self.q = self.s_q / (self.n_q + self._eps)

    def get_action(self, state):
        neighbours2newstate = {
        '000': '0',
        '001': '0',
        '010': '0',
        '011': '1',
        '100': '0',
        '101': '1',
        '110': '1',
        '111': '0',
        }
        prob = np.random.uniform(0.0, 1.0, 1)#1과 0사이에서 1개 선택
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            print("this is prob and epsilon",prob,self.epsilon)
            action = np.random.choice(range(self.num_actions))
            #목표 20개의 칸에서 가장 많이 살아있는공간을 출력하는것으로 하겠습니다.
            #아무것도 없는 경우 그 분야를 선택을 안하는 경우 >>0
            for i in range(4):
                self.x=sum(map(int,list(self.universe[5*i:5*(i+1)])))
                if self.prev<self.x:
                    self.prev=self.x
                    self.action=i
            print("this is action",action)
            self.prev=0
            # %3i: one number is converted into a field with a width of 3
            self.universe = self.offendvalue + self.universe + self.offendvalue
            self.universe = ''.join(neighbours2newstate[self.universe[i:i+3]] for i in range(self.cellcount))
        else:  #greedy
            action = self._policy_q[state, :].argmax()
        return action

    def improve_policy(self):
        self._policy_q = self.q.copy()
        #self.reset_statistics()

    def decaying_epsilon(self, factor):
        self.epsilon *= factor


class MCAgent(ExactMCAgent):
    """
    The 'learning-rate' Monte-Carlo agent.
    This agents performs value update as follows:
    V(s) <- V(s) + lr * (Gt - V(s))
    Q(s,a) <- Q(s,a) + lr * (Gt - Q(s,a))
    """

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(MCAgent, self).__init__(gamma=gamma,
                                      num_states=num_states,
                                      num_actions=num_actions,
                                      epsilon=epsilon)
        self.lr = lr

    def update(self, episode):
        states, actions, rewards = episode

        # reversing the inputs!
        # for efficient computation of returns
        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        iter = zip(states, actions, rewards)
        cum_r = 0
        for s, a, r in iter:
            cum_r *= self.gamma
            cum_r += r

            self.v[s] += self.lr * (cum_r - self.v[s])
            self.q[s, a] += self.lr * (cum_r - self.q[s, a])


def run_episode(env, agent, timeout=1000):
    env.reset()
    states = []
    actions = []
    rewards = []

    i = 0
    timeouted = False
    while True:
        state = env.s
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if done:
            break
        else:
            i += 1
            if i >= timeout:
                timeouted = True
                break

    if not timeouted:
        episode = (states, actions, rewards)
        agent.update(episode)


if __name__ == '__main__':
    from envs.gridworld import GridworldEnv

    nx, ny = 5, 5
    env = GridworldEnv([ny, nx])

    mc_agent = MCAgent(gamma=1.0,
                       lr=1e-3,
                       num_states=nx * ny,
                       num_actions=4,
                       epsilon=1.0)

    run_episode(env, mc_agent)
