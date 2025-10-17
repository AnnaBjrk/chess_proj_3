# Actor critic

# actor critic loss calculation td error och advantage
import math
import torch.distributions as dist
import torch


import torch
import torch.nn as nn


def plot_probabilities(probs):
    dist = Categorical(torch.tensor(probs))
    # Obtain the entropy in nats
    entropy = dist.____
    # Convert the entropy to bits
    entropy = entropy / math.log(____)
    print(
        f"{'Probabilities:':>15} {[round(prob, 3) for prob in dist.probs.tolist()]}")
    print(f"{'Entropy:':>15} {entropy:.2f}\n")
    plt.figure()
    plt.bar([str(x) for x in range(len(dist.probs))],
            dist.probs, color='skyblue', edgecolor='black')
    plt.ylabel('Probability')
    plt.xlabel('Action index')
    plt.ylim(0, 1)
    plt.show()


plot_probabilities([.25, .25, .25, .25])
plot_probabilities([.1, .15, .2, .25, .3])
# Try with your own list
plot_probabilities(____)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


# Usage example
action_probs = policy_network(state)
print('Action probabilities:', action_probs)

# Sample an action from the probability distribution
action_dist = torch.distributions.Categorical(action_probs)
action = action_dist.sample()

# Output example:
# Action probabilities: tensor([0.21, 0.02, 0.74, 0.03])


classCritic(nn.Module):


def __init__(self, state_size):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(state_size, 64)
    self.fc2 = nn.Linear(64, 1)
    defforward(self, state):
    x = torch.relu(self.fc1(torch.tensor(state)))
    value = self.fc2(x)
    return valuecritic_network = Critic(8)t


def calculate_losses(critic_network, action_log_prob,
                     reward, state, next_state, done):
    value = critic_network(state)
    next_value = critic_network(next_state)
    # Calculate the TD target
    td_target = (reward + gamma * next_value * (1-done))
    td_error = td_target - value
    # Calculate the actor loss
    actor_loss = -action_log_prob * td_error.detach()
    # Calculate the critic loss
    critic_loss = td_error ** 2
    return actor_loss, critic_loss


actor_loss, critic_loss = calculate_losses(
    critic_network, action_log_prob,
    reward, state, next_state, done
)
print(round(actor_loss.item(), 2), round(critic_loss.item(), 2))


# training a A2C algoritm

for episode in range(10):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    while not done:
        step += 1
        if done:
            break
        # Select the action
        action, action_log_prob = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        # Calculate the losses
        actor_loss, critic_loss = calculate_losses(
            critic, action_log_prob,
            reward, state, next_state, done)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        state = next_state
    describe_episode(episode, reward, episode_reward, step)

    # Clipped probability ratio räkna
    log_prob = torch.tensor(.5).log()
log_prob_old = torch.tensor(.4).log()


def calculate_ratios(action_log_prob, action_log_prob_old, epsilon):
    # Obtain prob and prob_old
    prob = action_log_prob.exp()
    prob_old = action_log_prob_old.exp()
    # Detach the old action log prob
    prob_old_detached = prob_old.detach()
    # Calculate the probability ratio
    ratio = prob / prob_old_detached
    # Apply clipping
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    print(f"+{'-'*29}+\n|         Ratio: {str(ratio)} |\n| Clipped ratio: {str(clipped_ratio)} |\n+{'-'*29}+\n")
    return (ratio, clipped_ratio)


_ = calculate_ratios(log_prob, log_prob_old, epsilon=.2)

# clipped surrogat loss function


def calculate_losses(critic_network, action_log_prob, action_log_prob_old,
                     reward, state, next_state, done):
    value = critic_network(state)
    next_value = critic_network(next_state)
    td_target = (reward + gamma * next_value * (1-done))
    td_error = td_target - value
    # Obtain the probability ratios
    ratio, clipped_ratio = calculate_ratios(
        action_log_prob, action_log_prob_old, epsilon=.2)
    # Calculate the surrogate objectives
    surr1 = ratio * td_error.detach()
    surr2 = clipped_ratio * td_error.detach()
    # Calculate the clipped surrogate objective
    objective = torch.min(surr1, surr2)
    # Calculate the actor loss
    actor_loss = -objective
    critic_loss = td_error ** 2
    return actor_loss, critic_loss


actor_loss, critic_loss = calculate_losses(critic_network, action_log_prob, action_log_prob_old,
                                           reward, state, next_state, done)
print(actor_loss, critic_loss)

# träna PPO algoritm
for episode in range(10):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    while not done:
        step += 1
        action, action_log_prob, entropy = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        actor_loss, critic_loss = calculate_losses(critic, action_log_prob, action_log_prob,
                                                   reward, state, next_state, done)
        # Remove the entropy bonus from the actor loss
        actor_loss -= 0.01 * actor_loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        state = next_state
    describe_episode(episode, reward, episode_reward, step)

    # A2C med batch updates

    actor_losses = torch.tensor([])
critic_losses = torch.tensor([])
for episode in range(10):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    while not done:
        step += 1
        action, action_log_prob = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        actor_loss, critic_loss = calculate_losses(
            critic, action_log_prob,
            reward, state, next_state, done)
        # Append to the loss tensors
        actor_losses = torch.cat((actor_losses, actor_loss))
        critic_losses = torch.cat((critic_losses, critic_loss))
        if len(actor_losses) >= 10:
            # Calculate the batch losses
            actor_loss_batch = actor_losses.mean()
            critic_loss_batch = critic_losses.mean()
            actor_optimizer.zero_grad()
            actor_loss_batch.backward()
            actor_optimizer.step()
            critic_optimizer.zero_grad()
            critic_loss_batch.backward()
            critic_optimizer.step()
            # Reinitialize the loss tensors
            actor_losses = torch.tensor([])
            critic_losses = torch.tensor([])
        state = next_state
    describe_episode(episode, reward, episode_reward, step)


# entropy bonus


def select_action(policy_network, state):
    action_probs = policy_network(state)
    action_dist = dist.Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    # Obtain the entropy of the policy
    entropy = action_dist.entropy()
    return (action.item(),
            log_prob.reshape(1),
            entropy)

# Actor loss


def calculate_actor_loss(log_prob, advantage, entropy, c_entropy):
    actor_loss = -(log_prob * advantage + c_entropy * entropy)
    return actor_loss

# Note: Categorical.entropy() is in nats; divide by math.log(2) for bits


# PPO training loop med entropy
for episode in range(10):
    state, info = env.reset()
    done = False
    while not done:
        action, action_log_prob, entropy = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        actor_loss, critic_loss = calculate_losses(critic, action_log_prob, action_log_prob,
                                                   reward, state, next_state, done)
        actor_loss -= c_entropy * entropy
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        state = next_state

# loop med batch training# Set rollout length
rollout_length = 10

# Initiate loss batches
actor_losses = torch.tensor([])
critic_losses = torch.tensor([])

for episode in range(10):
    state, info = env.reset()
    done = False
    while not done:
        action, action_log_prob = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        actor_loss, critic_loss = calculate_losses(critic, action_log_prob,
                                                   reward, state, next_state, done)

        # Append step loss to the loss batches
        actor_losses = torch.cat((actor_losses, actor_loss))
        critic_losses = torch.cat((critic_losses, critic_loss))

        # If rollout is full, update the networks
        if len(actor_losses) >= rollout_length:
            # Take the batch average loss with .mean()
            actor_loss_batch = actor_losses.mean()
            critic_loss_batch = critic_losses.mean()

            # Perform gradient descent
            actor_optimizer.zero_grad()
            actor_loss_batch.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss_batch.backward()
            critic_optimizer.step()

            # Reinitialize the batch losses
            actor_losses = torch.tensor([])
            critic_losses = torch.tensor([])

        state = next_state


# från claude


def gameloopold():
    if choice == 1:
        # både actor och critic tränas i samma modell men med olika heads
        model = ChessModelActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # Hyperparameters
        T_horizon = 2048  # Steps before update
        K_epochs = 4      # Update epochs
        gamma = 0.99
        gae_lambda = 0.95
        epsilon = 0.2

        actor_losses = torch.tensor([])
        critic_losses = torch.tensor([])
        # loop med batch training# Set rollout length
        rollout_length = 10
        for episode in range(10):
            state, info = env.reset()
            done = False
            episode_reward = 0
            step = 0
            while not done:
                step += 1
                action, action_log_prob, entropy = select_action(
                    actor, state)
                next_state, reward, terminated, truncated, _ = env.step(
                    action)
                episode_reward += reward
                done = terminated or truncated
                actor_loss, critic_loss = calculate_losses(critic, action_log_prob,
                                                           reward, state, next_state, done)

                # Append step loss to the loss batches
                actor_losses = torch.cat((actor_losses, actor_loss))
                critic_losses = torch.cat((critic_losses, critic_loss))

                # Remove the entropy bonus from the actor loss
                actor_loss -= 0.01 * actor_loss

                # If rollout is full, update the networks
                if len(actor_losses) >= rollout_length:
                    # Take the batch average loss with .mean()
                    actor_loss_batch = actor_losses.mean()
                    critic_loss_batch = critic_losses.mean()

                    # Perform gradient descent
                    actor_optimizer.zero_grad()
                    actor_loss_batch.backward()
                    actor_optimizer.step()

                    critic_optimizer.zero_grad()
                    critic_loss_batch.backward()
                    critic_optimizer.step()

                    # Reinitialize the batch losses
                    actor_losses = torch.tensor([])
                    critic_losses = torch.tensor([])
                state = next_state
            describe_episode(episode, reward, episode_reward, step)
