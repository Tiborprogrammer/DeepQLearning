import gym
from pathlib import Path

from Agent import Agent, run_episode, load_agent, evaluate_agent

env = gym.make("LunarLander-v2")
state = env.reset()

start_exploration_rate = 0.2
end_exploration_rate = 0

exp_change_factor = 0.995

episode = 0
training = True
agent_dir = "/Users/coding/Desktop/Developer/Python/LunarLander-v2"
Path(agent_dir).mkdir(parents=True, exist_ok=True)

value = input("Load agent y/n?")
if value.lower() == "y":
    agent = load_agent(agent_dir)
    print("You loaded the agent!")

else:
    agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.001, start_exploration_rate, "LunarLander-v2")
    print("Agent not loaded!")

while True:
    if training:
        if episode % 100 == 0:
            print(f"Episode: {episode}")
            evaluate_agent(env, agent, agent_dir, 10, episode)

        run_episode(env, agent, should_learn=True)

        agent.exploration_rate *= exp_change_factor
    else:
        evaluate_agent(env, agent, agent_dir, 10, episode, should_render=True)

    episode += 1
    agent.writer.add_scalar("Episode", episode, agent.global_step)
