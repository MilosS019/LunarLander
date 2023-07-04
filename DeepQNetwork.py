import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import array as arr

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, layer1_dims, layer2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.n_actions = n_actions
        self.entry_layer = nn.Linear(self.input_dims, self.layer1_dims)
        self.middle_layer = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.exit_layer = nn.Linear(self.layer2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def get_estimated_q_value(self, state):
        x = F.relu(self.entry_layer(state))
        x = F.relu(self.middle_layer(x))
        actions = self.exit_layer(x)

        return actions

    def save(self,path):
        T.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(T.load(path))
        self.eval()

    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, mem_size = 200000, eps_min=0.01, eps_dec = 0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.mem_size = mem_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.mem_pointer = 0

        self.state_memory = [[] for i in range(self.mem_size)]
        self.new_state_memory = [[] for i in range(self.mem_size)]
        self.reward_memory = [0 for i in range(self.mem_size)]
        self.action_memory = [0 for i in range(self.mem_size)]
        self.terminal_memory = [False for i in range(self.mem_size)]

        self.Q_network = DeepQNetwork(lr = self.lr, n_actions = n_actions, input_dims = input_dims, layer1_dims = 256, layer2_dims = 256)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_pointer % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_pointer += 1


    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = T.tensor(state).to(self.Q_network.device)
            actions = self.Q_network.get_estimated_q_value(state)
            return T.argmax(actions).item()
        else:
            return np.random.choice(self.action_space)
        

    def learn(self):
        if self.mem_pointer < self.batch_size:
            return
        
        self.Q_network.optimizer.zero_grad()

        max_mem = min(self.mem_pointer, self.mem_size)
        batch = [np.random.choice(max_mem) for i in range(self.batch_size)]
        batch_index = [i for i in range(self.batch_size)]

        state_memory_elements = [arr.array('d',self.state_memory[index]) for index in batch]
        new_state_memory_elements = [arr.array('d',self.new_state_memory[index]) for index in batch]
        reward_memory_elements = arr.array('d',[self.reward_memory[index] for index in batch])
        terminal_memory_elements = [int(self.terminal_memory[index]) for index in batch]


        state_batch = T.tensor(state_memory_elements).to(self.Q_network.device)
        new_state_batch = T.tensor(new_state_memory_elements).to(self.Q_network.device)
        reward_batch = T.tensor(reward_memory_elements).to(self.Q_network.device)
        terminal_batch = T.tensor(terminal_memory_elements).to(self.Q_network.device)


        action_batch = [self.action_memory[index] for index in batch]    

        Q_network = self.Q_network.get_estimated_q_value(state_batch)[batch_index, action_batch]
        q_next = self.Q_network.get_estimated_q_value(new_state_batch)
        q_target = reward_batch + self.gamma * (1 - terminal_batch) * T.max(q_next, dim = 1)[0]

        loss = self.Q_network.loss(q_target, Q_network).to(self.Q_network.device)    
        loss.backward()
        self.Q_network.optimizer.step()


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save(self, path):
        self.Q_network.save(path)

    def load(self, path):
        self.Q_network.load(path)