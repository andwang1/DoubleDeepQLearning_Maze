# Import some modules from other libraries
import numpy as np
import torch
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import collections

from environment import Environment
from q_visualisation import QVisualisation
from path_visualisation import PathVisualisation


# The Agent class allows the agent to interact with the environment.
class Agent:
    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, action: int = False):
        # Choose an action.
        if action is not False:
            discrete_action = action
        else:
            discrete_action = np.random.randint(0, 4)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this action.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = 1 - distance_to_goal
        return reward

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:  # Move left
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 2:  # Move up
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        else:  # Move down
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        return continuous_action


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:
    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function to calculate the loss for a single transition.
    def _calculate_loss(self, transition):
        current_state, action, reward, next_state = transition
        # Current state is 0x2, unsqueeze to convert to 1x2, as all functions need 2D tensors
        input_tensor = torch.tensor(current_state).unsqueeze(0)
        # Network prediction is a 1x4 tensor of 4 state value predictions, one for each action
        network_prediction = self.q_network.forward(input_tensor)
        # Turn action and reward into a 2D 1x1 tensor as gather and MSELoss take 2D tensors
        tensor_action_index = torch.tensor([[action]])
        reward_tensor = torch.tensor([[reward]])
        # Select for each 1x4 tensor of network predictions the 1x1 tensor related to the action in the transition
        # Gather is like indexing, 1 is the axis, pick the index in the column given by tensor_action_index
        # Output will always be the same dimension as the tensor_action_index, like masking
        predicted_q_for_action = torch.gather(network_prediction, 1, tensor_action_index)
        return torch.nn.MSELoss()(predicted_q_for_action, reward_tensor)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network_batch(self, transitions: tuple):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        tensor_current_states, tensor_actions, tensor_rewards, tensor_next_states = transitions
        # Network predictions is a *x4 tensor of 4 state value predictions per row, one for each action
        network_predictions = self.q_network.forward(tensor_current_states)
        predicted_q_values_for_action = torch.gather(network_predictions, 1, tensor_actions)
        loss = torch.nn.MSELoss()(predicted_q_values_for_action, tensor_rewards)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def return_optimal_action_order(self, input_tensor):
        network_prediction = self.q_network.forward(input_tensor)
        # Detach to remove the gradient component of the tensor, Numpy to convert to 2D np array, ravel to convert to 1D
        predictions_np_array = network_prediction.detach().numpy().ravel()
        # Normalise the predictions to a [0, 1] range to get the linear colour interpolation points
        colour_interpolation_factors = (predictions_np_array - min(predictions_np_array)) / (
                                        max(predictions_np_array) - min(predictions_np_array))
        return colour_interpolation_factors

    def return_greedy_action(self, current_state):
        input_tensor = torch.tensor(current_state).unsqueeze(0)
        network_prediction = self.q_network.forward(input_tensor)
        print(network_prediction)
        predictions_np_array = network_prediction.detach().numpy().ravel()
        return np.argmax(predictions_np_array)

class ReplayBuffer:
    def __init__(self, max_capacity=1000000):
        self.replay_buffer = collections.deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)

    # Returns tuple of tensors, each has dimension (batch_size, *), SARS'
    def generate_batch(self, batch_size=50):
        current_states = []
        actions = []
        rewards = []
        next_states = []
        indices = np.random.choice(range(len(self.replay_buffer)), batch_size, replace=False)
        for index in indices:
            transition = self.replay_buffer[index]
            current_states.append(transition[0])  # 1x2
            actions.append([transition[1]])  # 1x1
            rewards.append([transition[2]])  # 1x1
            next_states.append(transition[3])  # 1x2
        return torch.tensor(current_states), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(
            next_states)  # MSE needs float values, so cast rewards to floats

# Main entry point
if __name__ == "__main__":
    plot_loss = True
    plot_qvalues = False
    plot_state_path = False
    # Set the random seed for both NumPy and Torch
    CID = 741321
    np.random.seed(CID)
    torch.manual_seed(CID)
    # Create an environment.
    environment = Environment(display=False, magnification=1000)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    # Create a ReplayBuffer and batch size
    replay_buffer = ReplayBuffer()
    rb_batch_size = 50
    print("obstacle")
    print(dqn.return_greedy_action([0.35, 0.25]))

    # Loop over episodes
    episode_counter = 0
    losses = []
    time_steps = []
    initial_time = False
    while True:
        if episode_counter == 25:
            break
        episode_counter += 1
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode.
        for step_num in range(20):
            transition = agent.step()
            print(transition) # TODO
            print(dqn.return_greedy_action(transition[0]))
            # Skip the setup time to get as the first time for time plotting when the agent has made the first step.
            if initial_time is False:
                initial_time = datetime.now()
            replay_buffer.add(transition)
            if len(replay_buffer) < rb_batch_size:
                # Take time each step, even if we dont train
                time_steps.append(round((datetime.now() - initial_time).total_seconds() * 1000)) #time taken in milliseconds
                continue
            loss = dqn.train_q_network_batch(replay_buffer.generate_batch(rb_batch_size))
            # Measure time between steps (and training) in milliseconds for plotting
            time_steps.append(round((datetime.now() - initial_time).total_seconds() * 1000)) #time taken in milliseconds
            # losses.append(np.log10(loss)) # log loss
            losses.append(loss)  # abs loss
            # If want to display the environment slower after certain number of episode
            # if counter >= 15:
            #     time.sleep(0.5)

    # Plotting the loss functions as function of steps and time
    if plot_loss:
        time_steps = np.array(time_steps)
        time_steps = time_steps - time_steps[0]
        # print(len(time_steps))
        # print(len(losses))

        # Step axis
        ax1 = sns.lineplot(range(rb_batch_size, len(losses) + rb_batch_size), losses)
        ax1.set_xlabel("No. of steps")
        ax1.set_xticks(range(0, 501, 50))
        ax1.set_xlim([1, len(losses) + rb_batch_size - 1])  # make the x axis start at 1
        plt.yscale("log")
        # Turn off small ticks in between created by log
        plt.minorticks_off()
        plt.ylabel("Loss")
        plt.title("Batch Learning - Gamma = 0")
        # Time axis
        ax2 = ax1.twiny()
        time_labels_per_episode = [time_steps[i] for i in range(0, len(losses) + rb_batch_size - 1, rb_batch_size)]
        time_labels_per_episode.append(time_steps[-1])
        time_labels_positions = list(range(0, len(losses) + rb_batch_size, rb_batch_size))
        # time_labels_positions.append(len(losses) + rb_batch_size - 1)
        ax2.set_xticks(time_labels_positions)
        ax2.set_xticklabels(time_labels_per_episode)
        ax2.set_xlabel('Time (in ms)')
        ax2.set_xlim(ax1.get_xlim())

        # Add vertical lines
        for step_num in range(0, len(losses) + rb_batch_size, 20):
            ax1.axvline(step_num, ls="--", color="black", linewidth=0.2)
        plt.show()

    # steps of 0.05 as each state is 0.1 distance away, know from the obstacle
    if plot_qvalues:
        # # Because CV plots from top to bottom, origin is top left, we start with the upper row of states
        states_x_coords = np.arange(0.05, 1, 0.1)
        states_y_coords = np.arange(0.95, 0, -0.1)

        colour_factors = []
        for y_coord in states_y_coords:
            for x_coord in states_x_coords:
                input_tensor = torch.tensor([[x_coord, y_coord]])
                colour_factors.append(dqn.return_optimal_action_order(input_tensor))

        qv = QVisualisation(1000)
        qv.draw(colour_factors)
        time.sleep(15)

    if plot_state_path:
        state_path = []
        agent.reset()
        # Loop over steps within this episode.
        for step_num in range(20):
            # Take the greedy action step to plot the state path
            current_state = agent.state
            state_path.append(current_state)
            greedy_action = dqn.return_greedy_action(current_state)
            transition = agent.step(greedy_action)
            print(transition)

        pv = PathVisualisation(1000)
        pv.draw(state_path, True)
        time.sleep(15)