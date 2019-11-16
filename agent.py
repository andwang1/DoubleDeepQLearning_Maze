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

import time #TODO
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
        self.episode_length = 150 # 100 is the episode they will run at TODO SCALE WITH TIME?
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Batch size for replay buffer
        self.batch_size = 60
        # Replay buffer
        self.buffer_size = 3000
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        # Step size for each step
        self.step_length = 0.015  # TODO size of normalisation
        # DQN
        self.dqn = DQN(self.step_length, self.batch_size, replay_buffer_size=self.buffer_size)
        self.dqn.copy_weights_to_target_dqn()
        self.dqn.episode_length = self.episode_length
        # Share access to the same replay_buffer
        self.dqn.replay_buffer = self.replay_buffer

        self.random_exploration_epsilon = 1

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state: np.ndarray):
        # RANDOM EXPLORATION IN BEGINNING
        if self.num_steps_taken < self.episode_length * 5:
            action = self.dqn.test_current_state_actions[:, [2, 3]][np.random.randint(self.dqn.initial_sample_size)]
            action = np.array(action)
            # if self.num_steps_taken > self.episode_length * 2.5:
            #     self.random_exploration_epsilon -= 1 / self.episode_length
            #     print(self.random_exploration_epsilon)
            #     action = self.dqn.epsilon_greedy_policy(self.dqn.return_greedy_action(state), self.random_exploration_epsilon)
            # else:
            #     action = self.dqn.test_current_state_actions[:, [2, 3]][np.random.randint(self.dqn.initial_sample_size)]
            #     action = np.array(action)

        else:
            # PLUG DIRECTLY INTO HERE TO REDUCE FUNCTION CALLS
            action = self.dqn.epsilon_greedy_policy(self.dqn.return_greedy_action(state))

            # DEBUG SEE WHAT HAPPENS AT START AND END OF EPISODES
            # if self.num_steps_taken % self.episode_length <10:
            #     time.sleep(0.5)
            # if self.num_steps_taken % self.episode_length >90:
            #     time.sleep(0.5)


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
        # If stuck give negative reward
        if np.linalg.norm(self.state - next_state) < 0.002:
            # print("NOMOVE")
            reward = -.75 * distance_to_goal
        else:
            reward = 0.5 - distance_to_goal

        # types (list, np.float64, list)
        transition = (list(self.state) + self.action, reward, list(next_state))
        self.replay_buffer.add(transition)
        # Add new weight of 0 for the newest transition, we will make sure this gets picked manually by adding to batch
        # Cannot make this 0 for some reason will give error ValueError: probabilities contain NaN
        self.replay_buffer.transition_td_errors.append(0.0001)

        # Train
        if self.num_steps_taken > self.episode_length * 4.5:
            self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size), self.num_steps_taken)

        # CROSS ENTROPY METHOD
    def get_greedy_action(self, state: np.ndarray):
        return self.dqn.return_greedy_action(state)
        # call dqn greedy


# The DQN class determines how to train the above neural network.
class DQN:
    gamma = 0.9
    # The class initialisation function.
    def __init__(self, step_length=0.02, batch_size=50, replay_buffer_size=0, angles_between_actions=2):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=4, output_dimension=1)
        self.target_q_network = Network(input_dimension=4, output_dimension=1)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Step size for each step
        self.step_length = step_length  # TODO here decide whether to normalise and if what size of normalisation

        # Batch size used in the replay_buffer
        self.batch_size = batch_size # TODO update if change the batch size in replay buffer
        self.replay_buffer_size = replay_buffer_size # TODO FEED IN REPLAY BUFFER DIRECTLY

        # Sample size for initial sampling of actions to determine greedy action
        self.angles_between_actions = angles_between_actions
        # First (uniform) sample size
        self.initial_sample_size = int(360 / self.angles_between_actions)
        # Gaussian sample size
        self.gauss_sample_size = 40

        # Initialise arrays used to get greedy action in current state and greedy actions for all
        self.test_current_state_actions = False
        self.test_next_state_actions = False
        self.test_current_state_actions_gaussian = False
        self.test_next_state_actions_gaussian = False
        self.create_sample_test_steps()

        # Access to the same replay buffer
        self.replay_buffer = None

        # Epsilon
        self.epsilon = 1

        # Episode length
        self.episode_length = None

    # Creates an empty array with four columns, last 2 will be actions, split in angles
    # Input, how many degrees will be between each angle, i.e. 1, will give 360 actions
    def create_sample_test_steps(self):
        angles = np.array(np.arange(0, 360, self.angles_between_actions))
        x_steps = np.cos(angles) * self.step_length
        y_steps = np.sin(angles) * self.step_length

        self.test_current_state_actions = np.empty((self.initial_sample_size, 4))
        self.test_current_state_actions[:, 2] = x_steps
        self.test_current_state_actions[:, 3] = y_steps
        self.test_current_state_actions = torch.tensor(self.test_current_state_actions).float()

        # For each next state in the batch of transitions, we need to test num_of_samples actions to find the greedy action
        self.test_next_state_actions = np.empty((self.batch_size * self.initial_sample_size, 4))
        # Take the generated actions and write them into the empty np array to be used for testing
        self.test_next_state_actions[:, [2, 3]] = np.tile(self.test_current_state_actions[:, [2, 3]], reps=(self.batch_size, 1))
        self.test_next_state_actions = torch.tensor(self.test_next_state_actions).float()

        self.test_current_state_actions_gaussian = torch.empty((self.gauss_sample_size, 4))
        self.test_next_state_actions_gaussian = torch.empty((self.batch_size * self.gauss_sample_size, 4))

        # From the target network greedy action calculation we will return a tensor with the states selected by the batch and their corresponding greedy actions
        self.target_greedy_state_action_pairs = torch.empty((self.batch_size, 4))

        self.avg_td_error_at_start = None
        self.avg_td_error_at_end = None
        self.avg_td_error_mean = None
        self.avg_td_error_median = None

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
        # Update target network TODO HERE OR SOMEWHERE ELSE
        if step_number % 20 == 0:
            self.copy_weights_to_target_dqn()

        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        tensor_state_actions, tensor_rewards, tensor_next_states, buffer_indices = transitions
        # print(tensor_state_actions)
        # Network predictions is a *x1 tensor of the current state action values in the batch
        tensor_current_state_action_value = self.q_network.forward(tensor_state_actions)
        # print(tensor_current_state_action_value)
        # raise

        # Calculate the greedy next state action values in the batch
        with torch.no_grad():
            tensor_target_next_state_action_value = self.target_q_network.forward(self.return_next_state_q_greedy_target(transitions))

        # The bellman value of the current state action pairs in the batch
        tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_target_next_state_action_value
        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_current_state_action_value)
        loss.backward()
        # print(loss.item())
        # print(self.q_network.state_dict())
        self.optimiser.step()
        # print("bellman", tensor_bellman_current_state_value)
        # The absolute error between this bellman value and the current network's value for the same state action pairs in the batch
        td_error = np.abs((tensor_bellman_current_state_value - tensor_current_state_action_value).detach().numpy()).ravel()

        # Update the TD errors of the transitions we used
        for index, error in zip(buffer_indices, td_error):
            self.replay_buffer.transition_td_errors[index] = error

        # # Epsilon linear
        # self.epsilon -= 0.001

        # Avg uncertainty at start of this episode
        if step_number % self.episode_length == 10:
            self.avg_td_error_at_start = np.mean(td_error[-10:])

        # Avg uncertainty at end of the episode
        if step_number % self.episode_length == self.episode_length - 10:
            self.avg_td_error_at_end = np.mean(td_error[-10:])

        # Avg uncertainty over last episode
        if step_number % self.episode_length == 1:
            self.avg_td_error_mean = np.mean(td_error[-self.episode_length:])

        # Median uncertainty over last episode
        if step_number % self.episode_length == 1:
            self.avg_td_error_median = np.median(td_error[-self.episode_length:])

        # Set epsilon at start of episode
        if step_number % self.episode_length == 1:
            self.epsilon = 0.2
        # TODO CHANGE EPSILON CHANGE MAKE BIGGER< MAKE GAUSSIAN BIGGER
        # Make epsilon increase if growing uncertainty compared to start of episode whwer we are more greedy and should be precise
        if self.avg_td_error_mean:
            self.epsilon += 0.005 * (error - self.avg_td_error_mean) / self.avg_td_error_mean
            if self.epsilon < 0.05:
                self.epsilon = 0.05
            elif self.epsilon > 0.9:
                self.epsilon = 0.9


        # error will be the last error calculated which is the last one in the buffer

        # LAST ERROR DIVIDED BY MEAN? LAST ERROR WILL ALWAYS BE SMALLER BCS OF TARGET NETWORK, DO QNETWORK HERE INSTEAD?
        # COMPARE TO BEGINNING OF EPISODE?
        # self.last_transition_td_error = (error - min(self.replay_buffer.transition_td_errors)) / (max(self.replay_buffer.transition_td_errors) - min(self.replay_buffer.transition_td_errors))
        # self.epsilon = self.last_transition_td_error

        # EPSILON += fixed constant times sign (this error - previous error(mean of previous)) problem with target network
        # COMPUTE THE Qnetwork value instead?
        # start with low epsilon and increase over error uncertainty?

        # We take the first random exploration as benchmark to determine what randomness means in terms of TD errors
        # if self.first_batch_td is None:
        #     self.first_batch_td = np.mean(td_error)
        # # Latest error vs mean of first batch
        # else:
        #     self.epsilon = min(1, error / self.first_batch_td)



        # # CURRENTLY THIS IS SHRINKING AS THE TARGET NETWORK IS THE SAME WHEN WE RUN INTO A WALL
        # print(self.last_transition_td_error)

        # prints the value of the current state we are in, onlline
        # print(buffer_indices[-1])
        # print("qvalue", tensor_current_state_action_value[-1], tensor_state_actions[-1], tensor_rewards[-1])
        return loss.item()


        # TODO 2 use all TD ERRORS TO UPDATE WEIGHTS IN THE BATCH AND ALSO SET THE EPSILON FOR THE NEXT STEP
        # NEED TO SET THE WEIGHT FOR THE CURRENT TRANSITION equal to 1 so that it gets picked for sure and get the TD error for that, maybe append it lsat to the preformed batch? so batch size 119 plus current transition?
        # return indices of weight array from batch picking
        # Target action
        # target_actions

        # Given the next state, we want to find the greedy action in the next state and use it to compute the next state's value
        # This will now use the target_q_network

    def epsilon_greedy_policy(self, greedy_action, epsilon=False):
        if not epsilon:
            epsilon = self.epsilon
        # RANDOM
        print("eps", self.epsilon)
        if np.random.randint(0, 100) in range(int(epsilon * 100)):
            action = self.test_current_state_actions[:, [2, 3]][np.random.randint(self.initial_sample_size)]
            return np.array(action)
        # GREEDY
        else:
            return greedy_action

    def return_greedy_action(self, state: np.ndarray):
        # print("mag")
        # TODO EFFICIENCY APPEND VS STACK VS CONCAT, list append of lists vs numpy test efficiencey
        # combine to get stateaction tensor, np gives double by default so cast to float
        # Insert the current state into the predefined numpy array convert into float tensor
        # CROSS ENTROPY
        self.test_current_state_actions[:, [0, 1]] = torch.tensor(np.tile(state, (self.initial_sample_size, 1))) # ALIGN DATATYPES FROM BEGINNING TODO
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
        indices_highest_values = qvalues_tensor.argsort(axis=0)[-10:].squeeze()
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


    def return_next_state_q_greedy_target(self, transitions: torch.tensor): #TODO 1
        next_states = transitions[2]
        # print(next_states)
        # print(self.test_next_state_actions[:, [0, 1]].shape)

        # write states into test by repeating, so every state is matched with each action from the initial sample
        self.test_next_state_actions[:, [0, 1]] = np.repeat(next_states, self.initial_sample_size, axis=0)

        # write states into gaussian test by repeating, for later, the first gauss_sample_size rows will are the same state
        self.test_next_state_actions_gaussian[:, [0, 1]] = np.repeat(next_states, self.gauss_sample_size, axis=0)

        # print(self.test_next_state_actions[:80])
        # print(self.test_next_state_actions[140:160])
        # # raise
        with torch.no_grad():
            qvalues_tensor = self.target_q_network.forward(self.test_next_state_actions)

        # For every batch, get the actions corresponding to highest qvalues, resample and add into the gaussian test array
        gauss_start_index = 0
        gauss_end_index = self.gauss_sample_size
        for batch_end_index in range(self.initial_sample_size, self.batch_size * self.initial_sample_size + 1, self.initial_sample_size):
            batch_start_index = batch_end_index - self.initial_sample_size
            # Get the indices corresponding to the 20 best actions based on the qvalues
            indices_highest_values = qvalues_tensor[batch_start_index:batch_end_index].argsort(axis=0)[-20:].squeeze()
            best_actions = self.test_next_state_actions[:, [2, 3]][indices_highest_values].numpy()
            action_mean = np.mean(best_actions, axis=0)
            action_cov = np.cov(best_actions, rowvar=False)
            sampled_actions = np.random.multivariate_normal(action_mean, action_cov, size=(self.gauss_sample_size, 1)).reshape(-1, 2)
            sampled_actions = sampled_actions / (np.linalg.norm(sampled_actions, axis=1).reshape(-1, 1)) * self.step_length

            # Add sampled actions to the gaussian matrix
            self.test_next_state_actions_gaussian[gauss_start_index:gauss_end_index, [2, 3]] = torch.tensor(sampled_actions).float()
            gauss_start_index += self.gauss_sample_size
            gauss_end_index += self.gauss_sample_size

        # Second iteration
        greedy_actions = []
        with torch.no_grad():
            qvalues_tensor = self.target_q_network.forward(self.test_next_state_actions_gaussian)
        # print(qvalues_tensor)
        # argsort returns the indices from low to high, pick last 10 to get the 10 largest values
        for batch_end_index in range(self.gauss_sample_size, self.batch_size * self.gauss_sample_size + 1, self.gauss_sample_size):
            batch_start_index = batch_end_index - self.gauss_sample_size
            indices_highest_values = qvalues_tensor[batch_start_index:batch_end_index].argsort(axis=0)[-10:].squeeze()
            best_actions = self.test_next_state_actions_gaussian[:, [2, 3]][indices_highest_values].numpy()
            action_mean = np.mean(best_actions, axis=0)
            greedy_action = action_mean / np.linalg.norm(action_mean) * self.step_length
            greedy_actions.append(greedy_action)

        self.target_greedy_state_action_pairs[:, [0, 1]] = next_states
        self.target_greedy_state_action_pairs[:, [2, 3]] = torch.tensor(greedy_actions)
        # print("return target q")
        return self.target_greedy_state_action_pairs

        # gaussian (how most efficient) TODO EFFICIENCY
        # repeat (here have a precomputed empty array again, with the predefined gaussian sample sizes etc


class ReplayBuffer:
    def __init__(self, max_capacity, batch_size=50):
        self.buffer_max_len = max_capacity
        self.replay_buffer = collections.deque(maxlen=self.buffer_max_len)
        self.batch_size = batch_size
        self.length = 0
        # Initialise an array for the weights for each transition in the buffer, will make this a numpy array where
        # the indices will wrap around at the max_length to avoid performance degradation of resizing arrays
        # self.transition_weights = np.ones(max_capacity)
        self.transition_td_errors = collections.deque(maxlen=self.buffer_max_len)

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)
        # Adds 1 only if the buffer is not full
        self.length += 1 if self.length < self.buffer_max_len else 0

    def clear(self):
        self.replay_buffer.clear()

    # Returns tuple of tensors, each has dimension (batch_size, *), SARS'
    # TODO 3
    # TODO PICK SAMPLE BASED ON WEIGHT, DO WEIGHTS NEED TO BE STORED SEPERATELY T OKEEP USING THE DEQUE? seperate weights array use DEQUE so can keep the same length automatically as the buffer
    def generate_batch(self, batch_size = False):
        # REMOVE THIS IF NOT NEEDED, IE IF CONSTANT BATCH SIZE # TODO
        if not batch_size:
            batch_size = self.batch_size

        sum_of_errors = sum(self.transition_td_errors)
        # Adding a min probability constant to make sure transitions with small errors are still selected
        # TODO make it scale to exactly 0.01 percent based on length and sum
        min_probability_constant = 0.01 * sum_of_errors

        # print(self.transition_td_errors)
        # print(len(self.transition_td_errors))
        # print(self.length)
        # Normalise weights
        weights = (np.array(self.transition_td_errors) + min_probability_constant) / (
                    sum_of_errors + min_probability_constant * self.length)
        # weights = (np.array(self.transition_td_errors) + min_probability_constant) / (sum_of_errors + min_probability_constant * self.length)
        # print(weights)
        # weights =
        state_actions = []
        rewards = []
        next_states = []
        # We generate random indices according to their TD error weights
        indices = np.random.choice(range(self.length), batch_size, replace=False, p=weights)
        # We add the last transition to the buffer so it is trained on for sure, from this we will then get the TD error
        # We replace the last transition picked, this will likely have the lowest prob and be least important, we do
        # this because append is slow and creates a copy
        indices[-1] = self.length - 1
        for index in indices:
            transition = self.replay_buffer[index]
            state_actions.append(transition[0])  # 1x4 LIST
            rewards.append([transition[1]])  # 1x1
            next_states.append(transition[2])  # 1x2
        return torch.tensor(state_actions), torch.tensor(rewards).float(), torch.tensor(next_states), indices
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