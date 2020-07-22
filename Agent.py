import json
import random
from collections import deque
from datetime import datetime

import torch
import copy
from torch import nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter as SW


class ReplayBuffer:
    def __init__(self, max_size):
        self.transitions = deque(maxlen=max_size)

    def add(self, transition):
        self.transitions.append(transition)

    def sample(self, num_of_samples):
        samples = []
        for i in range(0, num_of_samples):
            samples.append(random.choice(self.transitions))

        return samples


class Agent:
    def __init__(self, state_dimension, num_actions, lr, start_exploration_rate, env_name, discount=0.99):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.env_name = env_name
        self.writer = SW(f"Logs/{env_name}/{current_time}")
        self.state_dimension = state_dimension
        self.num_actions = num_actions
        self.lr = lr
        self.exploration_rate = start_exploration_rate
        self.main_network = nn.Sequential(
            nn.Linear(state_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.optimiser = Adam(self.main_network.parameters(), lr=lr)
        self.target_network = copy.deepcopy(self.main_network)
        self.replay_buffer = ReplayBuffer(1000000)
        self.global_step = 0
        self.discount = discount

    def pick_action(self, state, should_learn):
        state = torch.tensor(state).float().unsqueeze(0)
        if random.random() < self.exploration_rate and should_learn:
            return random.randint(0, self.num_actions - 1)

        Q_vals_for_state = self.main_network(state)[0]
        best_Q_val = float("-inf")
        best_action = 0
        for i in range(self.num_actions):
            if best_Q_val < Q_vals_for_state[i]:
                best_Q_val = Q_vals_for_state[i]
                best_action = i
        return best_action

    def learn_from_transitions(self, states, actions, reward_for_actions, new_states, dones):
        actions = actions.unsqueeze(1)
        states = states.float()
        new_states = new_states.float()

        self.optimiser.zero_grad()

        best_action = self.main_network(new_states).argmax(dim=1, keepdim=True)
        val_for_next_state = self.target_network(new_states).gather(1, best_action).detach()
        estimate_q_vals = reward_for_actions + val_for_next_state * self.discount * (1 - dones)

        current_Q_values = self.main_network(states).gather(1, actions)
        current_Q_values = current_Q_values.squeeze()

        errors = (estimate_q_vals - current_Q_values) ** 2
        error = errors.mean()

        self.writer.add_scalar("Error", error, self.global_step)
        self.writer.add_scalar("Exploration rate", self.exploration_rate, self.global_step)

        error.backward()
        self.optimiser.step()

    def learn(self):
        transitions = self.replay_buffer.sample(100)
        states = []
        actions = []
        reward_for_actions = []
        new_states = []
        dones = []
        for state, action, reward_for_action, new_state, done in transitions:
            states.append(state)
            actions.append(action)
            reward_for_actions.append(reward_for_action)
            new_states.append(new_state)
            dones.append(done)

        self.learn_from_transitions(torch.tensor(states), torch.tensor(actions),
                                    torch.tensor(reward_for_actions), torch.tensor(new_states),
                                    torch.tensor(dones)
                                    )


def run_episode(env, agent: Agent, should_learn, should_render=False):
    steps_taken = 0
    total_reward = 0
    done = False
    state = env.reset()

    while not done:
        action = agent.pick_action(state, should_learn)

        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add((state, action, reward, next_state, int(done)))

        steps_taken += 1
        total_reward += reward
        if should_learn:
            agent.learn()
            agent.global_step += 1
        if should_render:
            env.render()
        state = next_state

    agent.target_network.load_state_dict(agent.main_network.state_dict())

    agent.writer.add_scalar("Training reward", total_reward, agent.global_step)
    agent.writer.add_scalar("Steps", steps_taken, agent.global_step)
    return (total_reward, steps_taken)


def save_agent(dir, agent: Agent):
    q_network_state = agent.main_network.state_dict()
    torch.save(q_network_state, dir + "/q_network_state.ckpt")

    agent_data = {"exploration_rate": agent.exploration_rate, "num_actions": agent.num_actions,
                  "lr": agent.lr, "state_dimension": agent.state_dimension, "env_name": agent.env_name,
                  "discount": agent.discount}

    with open(dir + "/agent.json", "w") as f:
        json.dump(agent_data, f)


def load_agent(dir):
    with open(dir + "/agent.json") as f:
        agent_data = json.load(f)
    q_network_state = torch.load(dir + "/q_network_state.ckpt")
    agent = Agent(agent_data["state_dimension"], agent_data["num_actions"],
                  agent_data["lr"], agent_data["exploration_rate"], agent_data["env_name"], agent_data["discount"])

    agent.main_network.load_state_dict(q_network_state)
    agent.target_network.load_state_dict(q_network_state)

    return agent


def evaluate_agent(env, agent, agent_dir, number_of_evals, episode, should_render=False):
    sum_of_reward = 0
    sum_of_steps = 0
    for i in range(0, number_of_evals):
        reward, steps_taken = run_episode(env, agent, should_learn=False, should_render=should_render)
        sum_of_reward += reward
        sum_of_steps += steps_taken

    average_reward = sum_of_reward / number_of_evals
    average_steps = sum_of_steps / number_of_evals
    print(f"Result for {number_of_evals} evaluations: ")
    print(f"Reward: {average_reward}   Steps taken: {average_steps}")
    print("---------------------------------")
    save_agent(agent_dir, agent)
    agent.writer.add_scalar("Reward", average_reward, episode)
