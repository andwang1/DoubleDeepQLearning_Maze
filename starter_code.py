# Import some modules from other libraries
import numpy as np
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import collections

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



    def return_optimal_action_order(self, input_tensor):
        network_prediction = self.q_network.forward(input_tensor)  # return tensor of 4 state value predictions, one for each action
        prediction_np_array = network_prediction.detach().numpy().ravel() # convert into a numpy array
        # print(prediction_np_array)
        # optimal_action_order = np.argsort(prediction_np_array) # take argsort, from left to right will be smallest to largest, detach to get rid of grad in tensor
        colour_interpolation_factors = (prediction_np_array - min(prediction_np_array)) / (max(prediction_np_array) - min(prediction_np_array))
        return colour_interpolation_factors
        # vectorise this into one call?


    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network_batch(self, transitions):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Create input tensor from batch inputs
        tensor_current_states, tensor_actions, tensor_rewards, tensor_next_states = transitions
        # print("rewards", tensor_rewards)
        network_prediction = self.q_network.forward(tensor_current_states)  # return tensor of 4 state value predictions, one for each action
        # print(network_prediction)
        predicted_q_values_for_action = torch.gather(network_prediction, 1, tensor_actions)
        # print(predicted_q_values_for_action)
        # print(tensor_rewards)
        # Calculate the loss for this transition.
        loss = torch.nn.MSELoss()(predicted_q_values_for_action, tensor_rewards)
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


class ReplayBuffer:
    def __init__(self, max_capacity=1000000):
        self.replay_buffer = collections.deque(maxlen=max_capacity)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)

    def __len__(self):
        return len(self.replay_buffer)

    # returns list of tuples containing the transitions
    def generate_batch(self, batch_size=50):
        current_states = []
        next_states = []
        actions = []
        rewards = []
        for _ in range(batch_size):
            transition = self.replay_buffer[np.random.randint(len(self))]
            current_states.append(transition[0]) # 1x2
            actions.append([transition[1]]) # 1x1
            rewards.append([transition[2]]) # 1x1
            next_states.append(transition[3]) # 1x2
        return torch.tensor(current_states), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(next_states) # MSE needs float values

        # return [self.replay_buffer[np.random.randint(len(self) + 1)] for _ in range(batch_size)]



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
    # Create a ReplayBuffer
    replay_buffer = ReplayBuffer()
    rb_batch_size = 50
    # Loop over episodes
    counter = 0 #TODO
    losses = []
    time_steps = []
    initial_time = False
    while True:
        if counter == 5:#TODO
            break
        counter +=1#TODO
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(20): # TODO
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            if initial_time is False:
                initial_time = time.time()
            replay_buffer.add(transition)
            if len(replay_buffer) < rb_batch_size:
                continue
            loss = dqn.train_q_network_batch(replay_buffer.generate_batch(rb_batch_size))
            time_steps.append(round((time.time() - initial_time) * 1000))  # time taken in milliseconds
            # losses.append(np.log(loss)) # y axis should have log scale
            losses.append(loss) # abs loss
            # if counter >= 15: # TODO
            #     time.sleep(0.5)

    if plot:
        time_steps = np.array(time_steps)
        time_steps = time_steps - time_steps[0]
        # print(len(losses))
        # print(len(losses) + rb_batch_size + 1)
        # print(rb_batch_size)
        ax1 = sns.lineplot(range(rb_batch_size, len(losses) + rb_batch_size), losses)
        ax1.set_xlim([rb_batch_size, len(losses) + rb_batch_size]) # make the x axis start at 1
        ax1.set_xlabel("No. of steps")
        plt.ylabel("log(loss)")

        # time axis
        ax2 = ax1.twiny()
        time_labels_per_episode = time_steps

        # print(time_labels_per_episode)
        time_labels_positions = range(0, len(losses) + rb_batch_size, rb_batch_size)
        ax2.set_xticks(time_labels_positions)
        ax2.set_xticklabels(time_labels_per_episode)
        ax2.set_xlabel('Time (ms)')
        ax2.set_xlim(ax1.get_xlim())

        for step_num in range(0, len(losses) + rb_batch_size, 20):
            ax1.axvline(step_num, ls="--")

        plt.show()

        # FIX BOTH GRAPHS y scales
        # start both x axis on 0?

    # steps of 0.05 as each state is 0.1 distance away, know from the obstacle


    states_x_coords = np.arange(0.05, 1, 0.1)
    # start from the top state, because cv plots from top to bottom, origin is top left
    states_y_coords = np.arange(0.95, 0, -0.1)

    colour_factors = []
    for y_coord in states_y_coords: # TODO [:1]
        for x_coord in states_x_coords:
            input_tensor = torch.tensor([[x_coord, y_coord]])
            colour_factors.append(dqn.return_optimal_action_order(input_tensor))

    from q_visualisation import QVisualisation

    qv = QVisualisation(1000)
    qv.draw(colour_factors)
    time.sleep(15)
    # input_tensor = torch.tensor([[0.05, 0.05]])
    # optimal_actions = dqn.return_optimal_action_order(input_tensor)
    # print(optimal_actions)





    # put all base triangles into a list and use the action indices to select
    # establish a colour for all and based on index assign the colour, also from list
    #


