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
        # Step size for each step
        self.step_length = 0.01  # TODO size of normalisation
        # DQN
        self.dqn = DQN(self.step_length)
        self.target_dqn = DQN(self.step_length)
        self.dqn.copy_weights_to_target_dqn()


    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state: np.ndarray):
        # RANDOM EXPLORATION IN BEGINNING
        if self.num_steps_taken < 100:
            action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
            action = action / np.linalg.norm(action) * self.step_length
        else:
            # PLUG DIRECTLY INTO HERE TO REDUCE FUNCTION CALLS
            action = self.get_greedy_action(state)


            # calculate epsilon
            # give the last transition a weight of 1 into the replay buffer, and then use that to index the transition
            # in the loss calculation, i.e get the loss for this step (take an intermediate step in calc loss before plugging into mse loss
            # save that error


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


        # CROSS ENTROPY METHOD
    def get_greedy_action(self, state: np.ndarray):
        return self.dqn.return_greedy_action(state)
        # call dqn greedy

    def train_network(self):
        loss = self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(50), self.num_steps_taken)  # CHANGE BATCH SIZE TODO
        
        # UPDATE TARGET NETWORK HERE TODO




# The DQN class determines how to train the above neural network.
class DQN:
    gamma = 0.9
    # The class initialisation function.
    def __init__(self, step_length=0.01, batch_size=50, angles_between_actions=2):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=4, output_dimension=1)
        self.target_q_network = Network(input_dimension=4, output_dimension=1)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Step size for each step
        self.step_length = step_length  # TODO here decide whether to normalise and if what size of normalisation

        # Batch size used in the replay_buffer
        self.batch_size = batch_size # TODO update if change the batch size in replay buffer

        # Sample size for initial sampling of actions to determine greedy action
        self.angles_between_actions = angles_between_actions
        # First (uniform) sample size
        self.sample_size = int(360 / self.angles_between_actions)
        # Gaussian sample size
        self.gauss_sample_size = 20

        # Initialise arrays used to get greedy action in current state and greedy actions for all
        self.test_current_state_actions = False
        self.test_next_state_actions = False
        self.test_current_state_actions_gaussian = False
        self.test_next_state_actions_gaussian = False
        self.create_sample_test_steps()

    # Creates an empty array with four columns, last 2 will be actions, split in angles
    # Input, how many degrees will be between each angle, i.e. 1, will give 360 actions
    def create_sample_test_steps(self):
        angles = np.array(np.arange(0, 360, self.angles_between_actions))
        x_steps = np.cos(angles) * self.step_length
        y_steps = np.sin(angles) * self.step_length

        self.test_current_state_actions = np.empty((self.sample_size, 4))
        self.test_current_state_actions[:, 2] = x_steps
        self.test_current_state_actions[:, 3] = y_steps
        self.test_current_state_actions = torch.tensor(self.test_current_state_actions).float()

        # For each next state in the batch of transitions, we need to test num_of_samples actions to find the greedy action
        self.test_next_state_actions = np.empty((self.batch_size * self.sample_size, 4))
        # Take the generated actions and write them into the empty np array to be used for testing
        self.test_next_state_actions[:, [2, 3]] = np.tile(self.test_current_state_actions[:, [2, 3]], reps=(self.batch_size, 1))
        self.test_next_state_actions = torch.tensor(self.test_next_state_actions).float()

        self.test_current_state_actions_gaussian = torch.empty((self.gauss_sample_size, 4))
        self.test_next_state_actions_gaussian = torch.empty((self.batch_size * self.gauss_sample_size, 4))

        # From the target network greedy action calculation we will return a tensor with the states selected by the batch and their corresponding greedy actions
        self.target_greedy_state_action_pairs = torch.empty((self.batch_size, 4))

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
    def train_q_network_batch(self, transitions: tuple, step_number):

        # Update target network
        if step_number % 50 == 0:
            self.copy_weights_to_target_dqn()

        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        tensor_state_actions, tensor_rewards, tensor_next_states = transitions
        # print(tensor_state_actions)
        # Network predictions is a *x1 tensor of a state action values
        network_predictions = self.q_network.forward(tensor_state_actions)

        with torch.no_grad():
            target_q_predictions = self.target_q_network.forward(self.return_next_state_q_greedy_target(transitions))

        tensor_bellman_current_state_value = tensor_rewards + self.gamma * target_q_predictions

        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, network_predictions)
        td_error = tensor_bellman_current_state_value - network_predictions

        print("bellman", tensor_bellman_current_state_value)
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()


        #TODO 2 use all TD ERRORS TO UPDATE WEIGHTS IN THE BATCH AND ALSO SET THE EPSILON FOR THE NEXT STEP
        # NEED TO SET THE WEIGHT FOR THE CURRENT TRANSITION equal to 1 so that it gets picked for sure and get the TD error for that, maybe append it lsat to the preformed batch? so batch size 119 plus current transition?
        # return indices of weight array from batch picking
        # Target action
        # target_actions

        # Given the next state, we want to find the greedy action in the next state and use it to compute the next state's value
        # This will now use the target_q_network


    def return_greedy_action(self, state: np.ndarray):
        print("mag")
        # TODO EFFICIENCY APPEND VS STACK VS CONCAT, list append of lists vs numpy test efficiencey
        # combine to get stateaction tensor, np gives double by default so cast to float
        # Insert the current state into the predefined numpy array convert into float tensor
        # CROSS ENTROPY
        self.test_current_state_actions[:, [0, 1]] = torch.tensor(np.tile(state, (self.sample_size, 1))) # ALIGN DATATYPES FROM BEGINNING TODO
        qvalues_tensor = self.q_network.forward(self.test_current_state_actions)
        # print(qvalues_tensor)
        # argsort returns the indices from low to high, pick last 20 to get the 20 largest values
        indices_highest_values = qvalues_tensor.argsort(axis=0)[-20:].squeeze()
        # print(indices_highest_values)
        # Get the best actions from that array, convert to numpy to use np mean function
        best_actions = self.test_current_state_actions[:, [2, 3]][indices_highest_values].numpy()
        # print("best", best_actions)
        action_mean = np.mean(best_actions, axis=0)
        # print(action_mean)

        # rowvar = False, tells numpy that columns are variables, and rows are samples, by default other way around
        action_cov = np.cov(best_actions, rowvar=False)

        # Sampling gives 3D matrix, reshape to 2D
        sampled_actions = np.random.multivariate_normal(action_mean, action_cov, size=(self.gauss_sample_size, 1)).reshape(-1, 2)
        # NEED TO TILE

        # Normalise sampled actions to step length
        sampled_actions = sampled_actions / (np.linalg.norm(sampled_actions, axis=1).reshape(-1, 1)) * self.step_length
        self.test_current_state_actions_gaussian[:, [0, 1]] = self.test_current_state_actions[:self.gauss_sample_size, [0, 1]]
        self.test_current_state_actions_gaussian[:, [2, 3]] = torch.tensor(sampled_actions).float()

        # Second iteration
        qvalues_tensor = self.q_network.forward(self.test_current_state_actions_gaussian)
        # print(qvalues_tensor)
        # argsort returns the indices from low to high, pick last 5 to get the 5 largest values
        indices_highest_values = qvalues_tensor.argsort(axis=0)[-5:].squeeze()
        # print(indices_highest_values)
        # Get the best actions from that array
        best_actions = self.test_current_state_actions_gaussian[:, [2, 3]][indices_highest_values].numpy()
        # print(best_actions)
        action_mean = np.mean(best_actions, axis=0)
        # print("mean", action_mean)
        greedy_action = action_mean / np.linalg.norm(action_mean) * self.step_length

        # print(np.linalg.norm(sampled_actions[0]))
        # print(np.linalg.norm(sampled_actions, axis=1))
        # print(sampled_actions)
        # print("samples")
        # print(sampled_actions)

        # print("state", state)
        # print("action", action_mean)
        # TODO SEE IF REQUIRED
        # If the max step size is exceeded scale it back
        # print("greedy action chosen")
        if np.linalg.norm(greedy_action) > 0.02:
            # action_mean *= 0.02 / np.linalg.norm(action_mean)
            print("STEP SIZE VIOLATED")
        return greedy_action


    def return_next_state_q_greedy_target(self, transitions: np.ndarray): #TODO 1
        next_states = transitions[2]
        # print(next_states)
        # print(self.test_next_state_actions[:, [0, 1]].shape)
        self.test_next_state_actions[:, [0, 1]] = torch.tensor(np.tile(next_states, (self.sample_size, 1)))
        with torch.no_grad():
            qvalues_tensor = self.target_q_network.forward(self.test_next_state_actions)

        # write states into gaussian test by repeating
        self.test_next_state_actions_gaussian[:, [0, 1]] = np.repeat(next_states, self.gauss_sample_size, axis=0)

        # here need to now do for every batch
        # actions_means = []
        # actions_covs = []
        gauss_start_index = 0
        gauss_end_index = gauss_start_index + self.gauss_sample_size
        for batch_end_index in range(self.sample_size, self.batch_size * self.sample_size + 1, self.sample_size):
            batch_start_index = batch_end_index - self.sample_size
            indices_highest_values = qvalues_tensor[batch_start_index:batch_end_index].argsort(axis=0)[-20:].squeeze()
            best_actions = self.test_next_state_actions[:, [2, 3]][indices_highest_values].numpy()
            action_mean = np.mean(best_actions, axis=0)
            # actions_means.append(action_mean)
            action_cov = np.cov(best_actions, rowvar=False)
            # actions_covs.append(action_cov)
            sampled_actions = np.random.multivariate_normal(action_mean, action_cov, size=(self.gauss_sample_size, 1)).reshape(-1, 2)
            sampled_actions = sampled_actions / (np.linalg.norm(sampled_actions, axis=1).reshape(-1, 1)) * self.step_length

            # Add sampled actions to the gaussian matrix
            self.test_next_state_actions_gaussian[gauss_start_index:gauss_end_index, [2, 3]] = torch.tensor(sampled_actions).float()
            gauss_start_index = gauss_end_index
            gauss_end_index += self.gauss_sample_size

        # Second iteration
        greedy_actions = []
        with torch.no_grad():
            qvalues_tensor = self.target_q_network.forward(self.test_next_state_actions_gaussian)
        # print(qvalues_tensor)
        # argsort returns the indices from low to high, pick last 5 to get the 5 largest values
        for batch_end_index in range(self.gauss_sample_size, self.batch_size * self.gauss_sample_size + 1, self.gauss_sample_size):
            batch_start_index = batch_end_index - self.gauss_sample_size
            indices_highest_values = qvalues_tensor[batch_start_index:batch_end_index].argsort(axis=0)[-20:].squeeze()
            best_actions = self.test_next_state_actions_gaussian[:, [2, 3]][indices_highest_values].numpy()
            action_mean = np.mean(best_actions, axis=0)
            greedy_action = action_mean / np.linalg.norm(action_mean) * self.step_length
            greedy_actions.append(greedy_action)

        self.target_greedy_state_action_pairs[:, [0, 1]] = next_states
        self.target_greedy_state_action_pairs[:, [2, 3]] = torch.tensor(greedy_actions)
        print("return target q")
        return self.target_greedy_state_action_pairs

        # use np array to take the next states from the batch and fill them into the big array
        # where the actions are preallocated (only once at initialisation), at each degree of half degree at same length = stepsize, circle with radius step size
        # then for each batch, pick the states from the batch and fill them into the preallocated numpy array (OR TENSOR)
        # plug that into forward, for each batch (precompute indices) pick the best actions (arg sort on slices one for each batch)
        # -----
        # gaussian (how most efficient)
        # repeat (here have a precomputed empty array again, with the predefined gaussian sample sizes etc
        pass


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
    # TODO 3
    # TODO PICK SAMPLE BASED ON WEIGHT, DO WEIGHTS NEED TO BE STORED SEPERATELY T OKEEP USING THE DEQUE? seperate weights array use DEQUE so can keep the same length automatically as the buffer
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
            state_actions.append(transition[0])  # 1x4 LIST
            rewards.append([transition[1]])  # 1x1
            next_states.append(transition[2])  # 1x2
        return torch.tensor(state_actions), torch.tensor(rewards).float(), torch.tensor(next_states)
        # MSE needs float values, so cast rewards to floats
        # next state needs to be appended with actions, do torch.cat(TUPLE(a,b)), will return new tensor


if __name__ == '__main__':
    dqn = DQN()
    dqn.create_sample_test_steps()
    print(dqn.test_current_state_actions)
    print(len(dqn.test_current_state_actions))
    print(len(dqn.test_next_state_actions))
    print(dqn.test_next_state_actions.shape)
    print(np.linalg.norm(dqn.test_next_state_actions[:,2:4], axis=1))