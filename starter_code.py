# Import some modules from other libraries
import numpy as np
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Import the environment module
from environment import Environment


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
    def step(self):
        # Choose an action.
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
        print(loss)
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

        # print(network_prediction)
        # output_for_action = network_prediction[action]
        # print(output_for_action)
        # np_action_index = np.zeros(4, dtype=int) # create an array to then populate the action we want to pick using torch gather of form Tensor([0, 0, 1, 0]) e.g.
        # np_action_index[action] = 1

        # tensor_action_index = torch.tensor([np_action_index]).long() # turn this into a tensor as gather takes tensors

        # print(action)
        # print(predicted_q_for_action)


# Main entry point
if __name__ == "__main__":
    plot = False
    # Set the random seed for both NumPy and Torch
    # You should leave this as 0, for consistency across different runs (Deep Reinforcement Learning is highly sensitive to different random seeds, so keeping this the same throughout will help you debug your code).
    CID = 1
    np.random.seed(CID)
    torch.manual_seed(CID)

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop PCs, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=1000)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()

    # Loop over episodes
    counter = 0 #TODO
    losses = []
    while True:
        if counter == 5:#TODO
            break
        counter +=1#TODO
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            loss = dqn.train_q_network(transition) # COMPUTES GRADIENT AND UPDATES WEIGHTS
            losses.append(loss)
            # if counter >= 15: # TODO
            #     time.sleep(0.5)

    if plot:
        ax = sns.lineplot(range(1, len(losses) + 1), losses)
        plt.xlim([1, len(losses) + 1]) # make the x axis start at 1
        for step_num in range(len(losses) + 1):
            if step_num % 20 == 0 and step_num != len(losses):
                ax.axvline(step_num, ls="--")

        plt.show()



