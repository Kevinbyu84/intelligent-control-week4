import gym
import numpy as np
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

for e in range(5):
    state, info = env.reset()  # Reset environment di awal episode
    state = np.reshape(state, [1, state_size])

    for time in range(500):
        env.render()  # Render environment (opsional)

        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Pastikan episode berhenti jika selesai

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:  # Jika episode selesai, hentikan loop
            print(f"Test Episode {e+1}, Score:{time}")
            break


env.close()