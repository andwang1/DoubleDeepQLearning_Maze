############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

""""NP CONCAT IS SLOW"""



import numpy as np
import torch
import collections
from path_visualisation import PathVisualisation


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


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 100 # 100 is the episode they will run at TODO
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Batch size for replay buffer
        self.batch_size = 50
        # Replay buffer
        self.replay_buffer = ReplayBuffer(1000000)
        # DQN
        self.dqn = DQN()
        self.target_dqn = DQN()
        self.dqn.copy_weights_to_target_dqn()


    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state):


        # RANDOM EXPLORATION IN BEGINNING
        if self.num_steps_taken > 100:
            action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        else:
            action = self.get_greedy_action(state)
            # print(action)
        # START TRAINING AFTER X STEPS


        # print("greedy")
        # print(state)
        # print(self.get_greedy_action(state))

        # CALC EPSILON
        # EPSILON GREEDY



        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition STORE AS LIST EFFICIENCY
        self.state = state # NP ARRAY
        # Store the action; this will be used later, when storing the transition
        self.action = list(action)
        return action # return here as nparray

    # AFTER ACTION CALL THIS GETS CALLED GET THE TRANSITION HERE TODO
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # HERE CHANGE THE REWARD TODO
        reward = 1 - distance_to_goal

        # BUNDLE STATEACTION INTO ONE ARRAY, NP APPEND IS SLOW, use list
        # types (list, np.float64, list)
        transition = (list(self.state) + self.action, reward, list(next_state))
        self.replay_buffer.add(transition)

        # NEED TO CALL TRAINING FROM HERE
        if self.num_steps_taken > 100:
            self.train_network()


    def get_greedy_action(self, state: np.ndarray):
        # start off with uniform


        # TODO MAKE STEPS SAME SIZE or restrict step size maximum test

        sampled_actions = np.random.uniform(low=-0.01, high=0.01, size=(10,2)).astype(np.float32) # HOW MANY SAMPLES TODO
        for _ in range(3): # how many resamplings TODO
            # combine to get stateaction tensor, np gives double by default so cast to float
            stateaction_tensor = torch.tensor(np.append(np.tile(state, (10, 1)), sampled_actions, axis=1)).float()
            # print(stateaction_tensor)
            qvalues_tensor = self.dqn.q_network.forward(stateaction_tensor)
            # print(qvalues_tensor)
            # argsort returns the indices from low to high, pick last 10 to get the 10 largest values
            indices_highest_values = qvalues_tensor.argsort(axis=0)[-10:].squeeze()
            # print(indices_highest_values)
            best_actions = sampled_actions[indices_highest_values]
            # print(best_actions)
            action_mean = np.mean(best_actions, axis=0)
            # print(action_mean)
            # HERE CAN EARLY BREAK ON LAST ITERATION once we have the mean
            # rowvar = False, tells numpy that columns are variables, and rows are samples by default reverse
            action_cov = np.cov(best_actions, rowvar=False)
            # print(action_cov)
            # Sampling gives 3D matrix, reshape to 2D
            sampled_actions = np.random.multivariate_normal(action_mean, action_cov, size=(10,1)).reshape(-1, 2)
            # print("samples")
            # print(sampled_actions)

        # print("state", state)
        # print("action", action_mean)
        # TODO SEE IF REQUIRED
        if np.linalg.norm(action_mean) > 0.02:
            action_mean *= 0.02 / np.linalg.norm(action_mean)
            print("STEP SIZE VIOLATED")
        return action_mean

    def train_network(self):
        loss = self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(50))  # CHANGE BATCH SIZE TODO
        
        # UPDATE TARGET NETWORK HERE TODO




# The DQN class determines how to train the above neural network.
class DQN:
    gamma = 0.9
    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=4, output_dimension=1)
        self.target_q_network = Network(input_dimension=4, output_dimension=1)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def copy_weights_to_target_dqn(self, other_dqn = False):
        if other_dqn:
            self.q_network.load_state_dict(other_dqn.q_network.state_dict())
        else:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

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
        tensor_state_actions, tensor_rewards, tensor_next_states = transitions
        # print(tensor_state_actions)
        # Network predictions is a *x1 tensor of a state action values
        network_predictions = self.q_network.forward(tensor_state_actions)
        # tensor_predicted_q_value_current_state = torch.gather(network_predictions, 1, tensor_actions)

        # Given the next state, we want to find the greedy action in the next state and use it to compute the next state's value
        # This will now use the target_q_network
        # tensor_next_states_values = self.return_next_state_values_tensor(tensor_next_states)
        # tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_next_states_values
        # loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_predicted_q_value_current_state)
        # # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        # loss.backward()
        # # Take one gradient step to update the Q-network.
        # self.optimiser.step()
        # Return the loss as a scalar
        # return loss.item()

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
        # print(int(network_prediction.argmax()))
        return int(network_prediction.argmax())

    def cross_entropy_greedy_action(self, state_tensor):
        pass


    def return_greedy_actions_tensor(self, tensor_states):
        tensor_network_predictions = self.q_network.forward(tensor_states)
        predictions_np_array = tensor_network_predictions.detach().numpy()
        greedy_actions = np.argmax(predictions_np_array, axis=1)
        # Reshape from 1D 1x* to 2D *x 1 array so can transform and output a tensor
        tensor_greedy_actions = torch.tensor(greedy_actions.reshape(-1, 1))
        return tensor_greedy_actions

    def return_next_state_values_tensor(self, tensor_next_states):
        # Using target network to predict the next state's values
        tensor_network_predictions = self.target_q_network.forward(tensor_next_states)
        predictions_np_array = tensor_network_predictions.detach().numpy()
        greedy_actions = np.argmax(predictions_np_array, axis=1)
        # Reshape from 1D 1x* to 2D *x 1 array so can transform and output a tensor
        tensor_greedy_actions = torch.tensor(greedy_actions.reshape(-1, 1))
        tensor_next_state_values = torch.gather(tensor_network_predictions, 1, tensor_greedy_actions)
        return tensor_next_state_values

class ReplayBuffer:
    def __init__(self, batch_size=50, max_capacity=1000000):
        self.replay_buffer = collections.deque(maxlen=max_capacity)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)

    def clear(self):
        self.replay_buffer.clear()

    # Returns tuple of tensors, each has dimension (batch_size, *), SARS'
    def generate_batch(self, batch_size = False):
        # REMOVE THIS IF NOT NEEDED, IE IF CONSTANT BATCH SIZE # TODO
        if not batch_size:
            batch_size = self.batch_size
        state_actions = []
        rewards = []
        next_states = []
        indices = np.random.choice(range(len(self.replay_buffer)), batch_size, replace=False) # ADD WEIGHTING TO BUFFER TODO
        for index in indices:
            transition = self.replay_buffer[index]
            state_actions.append(transition[0])  # 1x4
            rewards.append([transition[1]])  # 1x1
            next_states.append([transition[2]])  # 1x2
        return torch.tensor(state_actions), torch.tensor(rewards).float(), torch.tensor(next_states)
        # MSE needs float values, so cast rewards to floats
        # next state needs to be appended with actions, do torch.cat(TUPLE(a,b)), will return new tensor
