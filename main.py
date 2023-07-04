import gymnasium as gym
import numpy as np
import ast
import torch as T
import matplotlib.pyplot as plt
from QLearningLunarLanderAgent import QLearningLunarLanderAgent
from DeepQNetwork import Agent
    
def my_deep_q_learning():
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    state, info = env.reset(seed=2)
    episode_num = 600
    discovery_decay = 1 / episode_num * 1.25


    agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=4, eps_min = 0.01, input_dims = 8, lr=0.01, eps_dec=discovery_decay)
    # agent.load("learnedData.pth")
    scores, eps_history, avg_scores = [], [], [] 
    for index, _ in enumerate(range(episode_num)):
        state, info = env.reset(seed=1)
        score = 0
        while True:
            state, score, done = go_to_next_state(agent, score, env, state)
            if done:
                break
        
        agent.decrement_epsilon()
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print('episode', index, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' %agent.epsilon)
        if(avg_score >= 200):
            agent.save('fullyLearnedData.pth')
            break

    show_results(scores)
    show_results(avg_scores)
    agent.save('learnedData.pth')
    env.close() 


def go_to_next_state(agent, score, env, state):
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    score += reward
    next_state = next_state
    agent.store_transition(state, action, reward, next_state, terminated | truncated )

    agent.learn()
    if terminated or truncated:
        print("terminated: " + str(terminated) + ",   truncated: " + str(truncated) + ", reward: " + str(reward) + "\nselected actions: ")
        return next_state, score, True
    
    return next_state, score, False


def show_results(rewards):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


# def round_up(state,round_amount = 1):
#     for index,num in enumerate(state):
#         state[index] = round(num, round_amount)
#         # state[index] = int(num)

# def q_learning():
#     env = gym.make("LunarLander-v2", render_mode="human")
#     # env = gym.make("LunarLander-v2", render_mode="human")
#     state, info = env.reset(seed=1)
#     # q_table = {}
#     # load(q_table)
#     episode_num = 600
#     learning_rate = 0.1
#     discovery_rate = 1
#     discovery_decay = discovery_rate / episode_num
#     final_discovery_rate = 0
#     discount_factor = 0.95
#     rewards = []
#     agent = QLearningLunarLanderAgent(learning_rate, discovery_rate, discovery_decay, final_discovery_rate, discount_factor,env)
#     for index, _ in enumerate(range(episode_num)):
#         selected_action = {
#             0:0,
#             1:0,
#             2:0,
#             3:0
#         }
#         state, info = env.reset(seed=1)
#         # state = np.array(state)
#         round_up(state)
#         state = tuple(state)
#         episode_reward = 0
#         while True:
            
#             action = agent.get_action(state)
#             selected_action[action] += 1
#             next_state, reward, terminated, truncated, info = env.step(action)
#             episode_reward += reward
#             round_up(next_state)
#             next_state = tuple(next_state)
#             agent.update(state, action, reward, terminated, next_state)

#             if terminated or truncated:
#                 print("terminated: " + str(terminated) + ",   truncated: " + str(truncated) + ", reward: " + str(reward) + "\nselected actions: " + str(selected_action))
#                 print(str(agent.expolrationVsExploitation))
#                 break
#             state = next_state
        
#         rewards.append(episode_reward)
#         agent.decay_discovery_rate()
#         agent.resetExplorationVsExploitation()
#         print(index)

#     # show_results(rewards)
#     env.close() 

if __name__ == "__main__":
    my_deep_q_learning()