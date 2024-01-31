import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

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


# Configuration parameters for the whole setup
seed = 42
gamma = 0.9  # Discount factor for past rewards
max_steps_per_episode = 600
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
num_inputs = 1
num_actions = env.action_space.shape[0]
actions_array = np.linspace(env.action_space.low[0], env.action_space.high[0], num_actions)
num_hidden = 16

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

# training of actor critic model
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
episode_reward = 0
while True:  # Run until solved
    if episode_reward > 100:
        optimizer = keras.optimizers.Adam(learning_rate=0.1)

    state = env.reset()
    print(state)
    # state = state[0]
    infusion_arr = []
    episode_reward = 0
    doh_arr = []
    done_count = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # state = (100 - state)/100
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action_idx = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action = actions_array[action_idx]
            action_probs_history.append(tf.math.log(action_probs[0, action_idx]))

            # action = 0.6

            # Apply the sampled action in our environment
            state, reward, done = env.step(action)
            # print('timestep is ', timestep, ', reward is ', reward, ', and DoH is ', state, 'and action is', action)
            doh_arr.append(state)
            infusion_arr.append(action)
            rewards_history.append(reward)
            done_count += int(done)
            episode_reward += reward


        print(' Done count is ', done_count, ', reward is ', episode_reward, ', and DoH is ', state, 'and action is', action)
        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = (ret - value)
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append( #np.sqrt(diff**2))
                 huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
             )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    with summary_writer.as_default(step=episode_count):
        tf.summary.scalar('Episode Reward', episode_reward)

    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if done_count > max_steps_per_episode-300:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        figure, axis = plt.subplots(2)
        axis[0].plot(range(max_steps_per_episode), infusion_arr)
        axis[0].set_xlim(0, max_steps_per_episode)
        axis[0].set_ylim(0, 1.6)
        axis[0].set_ylabel('Infusion rate (mg/s)')

        axis[1].plot(range(max_steps_per_episode), doh_arr)
        axis[1].axhspan(45, 55, color='green', alpha=0.75, lw=0)
        # axis[1].plot(range(t_f), 45 * np.ones((t_f,)))
        axis[1].set_xlim(0, max_steps_per_episode)
        axis[1].set_ylabel("DoH (WAV_cns)")
        axis[1].set_xlabel("Time (seconds)")
        plt.show()
        break