# Import some modules from other libraries
import numpy as np
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import collections

# Import the environment module
from environment import Environment
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
        # Compute the reward for this paction.
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

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        pass
        # MSE ERROR OF PREDICTION AND REWARD
        state_array, action, reward, next_state = transition
        input_data = state_array # 0x2 array
        input_tensor = torch.tensor(input_data).unsqueeze(0) # CONVERT TO TORCH TENSOR, unsqueeze to convert from 0x2 to 1x2
        network_prediction = self.q_network.forward(input_tensor) # return tensor of 4 state value predictions, one for each action
        tensor_action_index = torch.tensor([[action]]) # turn action number into a 2D tensor as gather takes tensors
        predicted_q_for_action = torch.gather(network_prediction, 1, tensor_action_index) # select for each 1x4 tensor of predictions the 1x1 tensor related to the action in the transition
        reward_tensor = torch.tensor([[reward]]) # convert reward scalar into 1x1 tensor as MSELoss takes tensors
        return torch.nn.MSELoss()(predicted_q_for_action, reward_tensor)

    def return_greedy_action(self, current_state):
        input_tensor = torch.tensor(current_state).unsqueeze(0)
        network_prediction = self.q_network.forward(input_tensor)
        predictions_np_array = network_prediction.detach().numpy().ravel()
        print(predictions_np_array)  # DEBUG TODO
        return np.argmax(predictions_np_array)





        # print(network_prediction)
        # output_for_action = network_prediction[action]
        # print(output_for_action)
        # np_action_index = np.zeros(4, dtype=int) # create an array to then populate the action we want to pick using torch gather of form Tensor([0, 0, 1, 0]) e.g.
        # np_action_index[action] = 1

        # tensor_action_index = torch.tensor([np_action_index]).long() # turn this into a tensor as gather takes tensors

        # print(action)
        # print(predicted_q_for_action)


class ReplayBuffer:
    def __init__(self, max_capacity=1000000):
        self.replay_buffer = collections.deque(maxlen=max_capacity)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)

    def __len__(self):
        return len(self.replay_buffer)

    # returns list of tuples containing the transitions
    def generate_batch(self, batch_size=50):
        return [self.replay_buffer[np.random.randint(len(self) + 1)] for _ in range(batch_size)]



# Main entry point
if __name__ == "__main__":
    plot_loss = True
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
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            if initial_time is False:
                initial_time = time.time()
            loss = dqn.train_q_network(transition) # COMPUTES GRADIENT AND UPDATES WEIGHTS
            time_steps.append(round((time.time() - initial_time) * 1000)) #time taken in milliseconds
            losses.append(np.log10(loss)) # y axis should have log scale
            # losses.append(loss)  # abs loss
            # if episode_counter >= 15: # TODO
            #     time.sleep(0.5)

    # Reset so time starts at 0, take the time equal to 0 before the first training
    time_steps = np.array(time_steps)
    time_steps = time_steps - time_steps[0]
    rb_batch_size = 50

    if plot_loss:
        ax1 = sns.lineplot(range(len(losses)), losses)
        ax1.set_xlim([0, len(losses) + 1]) # make the x axis start at 0
        ax1.set_xlabel("No. of steps")
        ax1.set_xticks(range(0, 501, 50))
        plt.ylabel("Log(loss)")

        # time axis
        ax2 = ax1.twiny()
        time_labels_per_episode = [time_steps[i] for i in range(0, len(losses), rb_batch_size)]
        time_labels_per_episode.append(time_steps[-1])
        time_labels_positions = [i for i in range(0, len(losses), rb_batch_size)]
        time_labels_positions.append(len(losses))
        ax2.set_xticks(time_labels_positions)
        ax2.set_xticklabels(time_labels_per_episode)
        ax2.set_xlabel('Time (in ms)')
        ax2.set_xlim(ax1.get_xlim())

        for step_num in range(0, len(losses), 20):
            if step_num == len(losses):
                break
            ax1.axvline(step_num, ls="--", color="black", linewidth=0.2)

        plt.show()

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
            # print(transition) # DEBUG TODO

        pv = PathVisualisation(1000)
        pv.draw(state_path, True, True)
        time.sleep(15)