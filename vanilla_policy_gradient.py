
# Disable excessive info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import packages
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import numpy as np

# Turn off eager execution
tf.compat.v1.disable_eager_execution()


# --- SET UP ENVIRONMENT --- #


# Choose environment to use
env = gym.make('CartPole-v0')
num_actions = env.action_space.n
obs_size = len(env.observation_space.low)


# --- BUILD THE POLICY NETWORK --- #


# Input layer - 4 inputs to match env
inputs = tf.keras.Input(shape=(obs_size,))

# First hidden layer - 64 units, tanh activation
h1 = tf.keras.layers.Dense(
    units=32,
    activation=tf.keras.activations.tanh
)(inputs)

# Output layer - 2 outputs, identity activation
outputs = tf.keras.layers.Dense(
    units=num_actions,
    activation=tf.keras.activations.softmax
)(h1)

# Create a create optimizer and compile
policy_net = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(lr=1e-2)


# --- FUNCTIONS FOR TRAINING --- #

# Feed observation through policy network to get probabilities for actions
def get_policy(s):
    return policy_net.predict(s)[0]


# Feed observation through policy network and sample an action from output
def get_action(s):
    return int(np.random.choice(num_actions, 1, p=get_policy(s)))


# Loss function whose gradient is the policy gradient
# A: output from the network
# Y: 2D tensor (row 1 = actions, row 2 = returns) for each input state
def compute_loss(y, policy_output):

    # Extract data from Y
    acts = tf.cast(tf.gather(y, 0), tf.int32)
    rets = tf.cast(tf.gather(y, 1), tf.float32)

    # Get the probs for the actions taken
    mask = tf.one_hot(acts, num_actions)
    prob = tf.reduce_sum(policy_output * mask, axis=1)

    # Compute loss and return
    log_prob = tf.math.log(prob)
    return -tf.reduce_mean(log_prob * rets)


# --- COMPILE NETWORK --- #
policy_net.compile(optimizer=optimizer, loss=compute_loss)


# --- TRAINING LOOP --- #

epochs = 50
batch_size = 5000

for epoch in range(epochs):

    # Vars that reset for each batch
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []
    batch_lens = []

    # Vars that reset for each episode
    obs = env.reset()
    done = False
    ep_rew = []

    # Render first episode of each epoch
    rendered_this_epoch = False

    # Continue until batch complete
    while True:

        # Render if we're supposed to
        if not rendered_this_epoch:
            env.render()

        # Record new observation
        batch_obs.append(obs.tolist())

        # Choose a new action using policy network
        act = get_action(np.reshape(obs, (1, obs_size)).astype(np.float32))
        obs, rew, done, _ = env.step(act)

        # Record action and reward
        batch_acts.append(act)
        ep_rew.append(rew)

        # Check if episode complete
        if done:

            # Record data from the episode
            ep_ret = sum(ep_rew)
            ep_len = len(ep_rew)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_weights += [ep_ret] * ep_len

            # Reset the game
            obs = env.reset()
            done = False
            ep_rew = []

            # Don't render again this epoch
            rendered_this_epoch = True

            # Check if batch is finished
            if len(batch_obs) > batch_size:
                break

    # Pack observations, actions, and returns into tensors
    A = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
    Y1 = tf.convert_to_tensor(batch_acts, dtype=tf.float32)
    Y2 = tf.convert_to_tensor(batch_weights, dtype=tf.float32)
    Y = tf.stack([Y1, Y2])

    # Update network
    policy_net.train_on_batch(A, Y)

    # Print info
    info_str = "Epoch " + str(epoch) + " Complete:\t"
    info_str += "Avg. Return: " + str(sum(batch_rets) / len(batch_rets))
    info_str += "\tEpisodes: " + str(len(batch_rets))
    print(info_str)


