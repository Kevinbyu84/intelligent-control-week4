import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Replay Buffer
        self.memory = deque(maxlen=2000)

        # Main Network (Online Network)
        self.model = self._build_model()

        # Target Network (Salinan dari Online Network)
        self.target_model = clone_model(self.model)
        self.update_target_network()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        """Salin bobot dari online network ke target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Eksplorasi (aksi acak)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Eksploitasi (aksi terbaik)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)

            if done:
                target[0][action] = reward
            else:
                # Prediksi Q-value dari Target Network
                future_q = np.amax(self.target_model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + self.gamma * future_q

            # Latih model utama (online network)
            self.model.fit(state, target, epochs=1, verbose=0)

        # Kurangi epsilon untuk mengurangi eksplorasi
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent_on_env(env_name, episodes=300, target_update_frequency=10):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print(f"Environment: {env_name}, Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.4f}")
                break

            agent.replay()

        # Perbarui Target Network setiap beberapa episode
        if e % target_update_frequency == 0:
            agent.update_target_network()

    env.close()

if __name__ == '__main__':
    environments = ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0']

    for env_name in environments:
        print(f"\nTraining on {env_name}...")
        train_agent_on_env(env_name)
