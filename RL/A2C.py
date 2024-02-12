import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import anesthesia_simulation as asim

log_dir = "log/episode_rewards"
summary_writer = tf.summary.create_file_writer(log_dir)
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

t_s = 1
age = 45
weight = 64
height = 171
lbm = 52
env = asim.AnestheisaEnv(age, weight, height, lbm, t_s)


class Actor2Critic(Model):
    def __init__(self, num_hidden=16, num_actions=1):
        """
        :param num_hidden: 16 according to the paper
        :param num_actions: 1 drug infusion rate
        """
        super().__init__()
        self.common = layers.Dense(num_hidden, activation='relu')
        self.actor = layers.Dense(num_actions, activation='softmax')
        self.critic = layers.Dense(1, activation='linear')

    def call(self, state):
        """
        :param state: env.observation
        :return: action probability, critic value
        """
        x = self.common(state)
        return self.actor(x), self.critic(x)


class RLAgent:
    def __init__(self, num_hidden=16, num_action=1, max_steps_per_episode=600):
        self.model = Actor2Critic(num_hidden, num_action)
        self.max_steps_per_episode = max_steps_per_episode
        self.buffer = []
        self.num_action = num_action
        self.episode_reward = 0
        self.infusion_arr = []
        self.doh_arr = []
        self.done_count = 0

    def run_episode(self):
        state = env.reset()
        print(state)
        action_probs_history = []
        critic_value_history = []
        rewards_history = []
        self.infusion_arr.clear()
        self.doh_arr.clear()
        episode_reward = 0
        self.done_count = 0
        actions_array = np.linspace(env.action_space.low[0], env.action_space.high[0], self.num_action)
        for timestep in range(1, self.max_steps_per_episode):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = self.model.call(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action_idx = np.random.choice(self.num_action, p=np.squeeze(action_probs))
            action = actions_array[action_idx]
            action_probs_history.append(tf.math.log(action_probs[0, action_idx]))

            # Apply the sampled action in our environment
            state, reward, done = env.step(action)
            self.doh_arr.append(state)
            self.infusion_arr.append(action)
            rewards_history.append(reward)
            self.done_count += int(done)
            episode_reward += reward

        print(' Done count is ', self.done_count, ', episode reward is ', episode_reward, ', and DoH is ', state, 'and action is', action)
        self.episode_reward = episode_reward
        return action_probs_history, critic_value_history, rewards_history

    def remember(self, action_probs_history, critic_value_history, return_history, buffer_size):
        buffer_extend = zip(action_probs_history, critic_value_history, return_history)
        self.buffer.extend(buffer_extend)
        if len(self.buffer) > buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        action_probs, critic_value, rewards = zip(*[self.buffer[i] for i in batch])
        return action_probs, critic_value, rewards

    def train(self, max_episode, gamma, eps):
        huber_loss = keras.losses.Huber()
        batch_size = 3000
        buffer_size = 30000
        optimizer = keras.optimizers.Adam(learning_rate=0.05)
        # training of actor critic model
        for episode in range(max_episode):
            with tf.GradientTape() as tape:
                action_probs_episode, critic_value_episode, rewards_episode = self.run_episode()

                # Calculate expected value from rewards
                returns = []
                discounted_sum = 0
                for r in rewards_episode[::-1]:
                    discounted_sum = r + gamma * discounted_sum
                    returns.insert(0, discounted_sum)
                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                self.remember(action_probs_episode, critic_value_episode, returns, buffer_size)
                action_probs_history, critic_value_history, return_history = self.sample_batch(batch_size)

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, return_history)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    diff = (ret - value)
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(  # np.sqrt(diff**2))
                        huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )
                # Backpropagation
                cl_sum = sum(critic_losses)
                al_sum = sum(actor_losses)
                loss_value = cl_sum + al_sum
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Log details
            with summary_writer.as_default(step=episode):
                tf.summary.scalar('loss_value', loss_value)

            if self.done_count > 250:
                self.model.save_weights(f'./training/checkpoint_episode{episode}')
                file_name = os.path.join('checkpoint_response', f"state_action_episode{episode}.csv")
                data = list(zip(self.infusion_arr, self.doh_arr))
                df = pd.DataFrame(data, columns=['Infusion rate', 'DoH'])
                df.to_csv(file_name)


def main():
    num_action = env.action_space.shape[0]
    num_hidden = 16
    max_steps_per_episode = 600
    # Configuration parameters for the whole setup
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    gamma = 0.9  # Discount factor for past rewards
    max_episode = 1000
    agent = RLAgent(num_hidden, num_action, max_steps_per_episode)
    agent.train(max_episode, gamma, eps)


if __name__ == "__main__":
    main()
