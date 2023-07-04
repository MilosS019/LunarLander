import numpy as np

class QLearningLunarLanderAgent:
    def __init__(self, learning_rate, discovery_rate, discovery_decay, final_discovery_rate, discount_factor, env, q_table = {}) -> None:
        self.learning_rate = learning_rate
        self.discovery_rate = discovery_rate
        self.discovery_decay = discovery_decay
        self.final_discovery_rate = final_discovery_rate
        self.discount_factor = discount_factor
        self.env = env
        self.expolrationVsExploitation = {"exploration" : 0,
                                          "exploitation" : 0}
        self.q_table = q_table

    def get_action(self, state):
        if np.random.random() < self.discovery_rate:
            self.expolrationVsExploitation["exploration"] += 1
            return self.env.action_space.sample()
        else:
            self.expolrationVsExploitation["exploitation"] += 1
            if state in self.q_table:
                return np.argmax(self.q_table[state])
            else:
                return self.env.action_space.sample()
        
    def update(self, state, action, reward, terminated, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0,0,0,0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0,0,0,0]
        future_q_value = (not terminated) * np.max(self.q_table[next_state])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * temporal_difference

    def decay_discovery_rate(self):
        self.discovery_rate = max(self.final_discovery_rate, self.discovery_rate - self.discovery_decay)
        # self.discovery_rate = self.discovery_rate * self.discovery_decay 
    
    def resetExplorationVsExploitation(self):
        self.expolrationVsExploitation = {"exploration" : 0,
                                          "exploitation" : 0}