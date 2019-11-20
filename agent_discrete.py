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
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=200) #OVERFITTING?
        self.layer_2 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_3 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_4 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_5 = torch.nn.Linear(in_features=200, out_features=200)
        self.output_layer = torch.nn.Linear(in_features=200, out_features=output_dimension)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.layer_5.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.leaky_relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.leaky_relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.leaky_relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.leaky_relu(self.layer_4(layer_3_output))
        layer_5_output = torch.nn.functional.leaky_relu(self.layer_5(layer_4_output))
        output = self.output_layer(layer_5_output)
        return output


class Agent:
    # Function to initialise the agent
    def __init__(self):
        # Replay buffer batch size
        self.batch_size = 50
        # Set the episode length (you will need to increase this)
        self.episode_length = 200 # 100 is the episode they will run at TODO SCALE WITH TIME?
        self.actual_episode_length = self.episode_length
        self.episode_counter = 0

        # Set random exploration episode length
        self.random_exploration_episode_length = 400
        self.steps_made_in_exploration = self.random_exploration_episode_length * 6

        self.training_threshhold = self.random_exploration_episode_length

        self.num_steps_taken = 0
        self.state = None
        self.action = None

        # Replay buffer
        # self.buffer_size = self.steps_made_in_exploration + self.random_exploration_episode_length
        self.buffer_size = 400000 # TODO
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        # DQN
        self.dqn = DQN(self.batch_size, replay_buffer_size=self.buffer_size)
        self.dqn.copy_weights_to_target_dqn()
        self.dqn.episode_length = self.episode_length
        self.dqn.steps_copy_target = self.episode_length
        self.dqn.steps_made_in_exploration = self.steps_made_in_exploration
        self.dqn.replay_buffer = self.replay_buffer

        self.got_stuck = False

        # 8 actions
        self.actions = np.array([[-0.02, 0],
                                 [-0.01414, -0.01414],
                                 [0, -0.02],
                                 [0.01414, -0.01414],
                                 [0.02, 0],
                                 [0.01414, 0.01414],
                                 [0, 0.02],
                                 [-0.01414, 0.01414],
                                 ])

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episode_counter += 1
            self.steps_taken_in_episode = 0
            self.got_stuck = False
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state: np.ndarray):   # TODO REMOVE IS GREEDY FROM RETURN
        # RANDOM EXPLORATION IN BEGINNING
        is_greedy = False # TODO REMOVE IS GREEDY FROM RETURN
        # while we are in the number of steps try every direction and if get stuck in less than 8 end episode
        if self.num_steps_taken < self.steps_made_in_exploration:
            self.episode_length = self.random_exploration_episode_length
            direction = (self.episode_counter - 1) % 8
            # IF GET STUCK EARLY THEN WE ARE NEXT TO WALL, quit this random exploration early
            if self.steps_taken_in_episode < 9 and self.got_stuck:
                self.repeat_episode = False
                self.episode_length = self.num_steps_taken + 1
                action = np.random.randint(8)
            elif self.steps_taken_in_episode < 20 and not self.got_stuck:
                action = direction
            else:
                action = np.random.randint(8)
        else:
            self.episode_length = self.actual_episode_length
            action, is_greedy = self.dqn.epsilon_greedy_policy(self.dqn.return_greedy_action(state)) # TODO REMOVE IS GREEDY FROM RETURN

        self.steps_taken_in_episode += 1
        self.num_steps_taken += 1
        self.state = state # NP ARRAY
        self.action = action
        action = np.array(self.actions[action])
        return action, is_greedy # return here as nparray # TODO REMOVE IS GREEDY FROM RETURN

    # AFTER ACTION CALL THIS GETS CALLED GET THE TRANSITION HERE TODO
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if np.linalg.norm(self.state - next_state) < 0.0002:
            self.got_stuck = True
        else:
            self.got_stuck = False
        if distance_to_goal < 0.01:
            reward = 200
        elif distance_to_goal < 0.03:
            reward = 100
        elif distance_to_goal < 0.05:
            reward = 20
        elif distance_to_goal < 0.1:
            reward = 10
        elif distance_to_goal < 0.2:
            reward = 7
        elif distance_to_goal < 0.3:
            reward = 5
        elif distance_to_goal < 0.4:
            reward = 4
        elif distance_to_goal < 0.5:
            reward = 3
        elif distance_to_goal < 0.6:
            reward = 2
        elif distance_to_goal < 0.7:
            reward = 1
        elif distance_to_goal < 0.8:
            reward = 0.5
        else:
            reward = 0
        if reward > 0:
            print("reward", reward)

        transition = (self.state, self.action, reward, next_state)
        self.replay_buffer.add(transition)
        # Add new weight of 1 for the newest transition, we will make sure this gets picked manually by adding to batch
        # Cannot make this 0 for some reason will give error ValueError: probabilities contain NaN
        self.replay_buffer.transition_td_errors.append(0.0001)

        # Train
        if self.num_steps_taken > self.training_threshhold:
            self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size), self.num_steps_taken, self.got_stuck)

    def get_greedy_action(self, state: np.ndarray):
        return self.actions[self.dqn.return_greedy_action(state)]


# The DQN class determines how to train the above neural network.
class DQN:
    gamma = .95
    # The class initialisation function.
    def __init__(self, batch_size, replay_buffer_size):
        self.q_network = Network(input_dimension=2, output_dimension=8)
        self.target_q_network = Network(input_dimension=2, output_dimension=8)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Batch size used in the replay_buffer
        self.replay_buffer = None
        self.batch_size = batch_size # TODO update if change the batch size in replay buffer
        self.replay_buffer_size = replay_buffer_size # TODO FEED IN REPLAY BUFFER DIRECTLY

        # Episode length
        self.episode_length = None
        self.episode_counter = 0
        self.steps_copy_target = self.episode_length

        # Epsilon
        self.epsilon = 0.8 # TODO
        self.steps_increase_epsilon = 15

        # is greedy
        self.is_greedy = False
        self.epsilon_maxed = False

        # # Epsilon linear in episode length
        self.is_epsilon_delta = True
        self.is_epsilon_greedy = False

        self.start_epsilon_delta = 0.5
        self.start_epsilon_greedy = 0.2

        self.epsilon_decrease = 0.0002
        self.epsilon_increase = 0.003

        self.steps_made_in_exploration = 0
        self.greedy_counter = 0


    def train_q_network_batch(self, transitions: tuple, step_number, got_stuck):
        # Update target network
        if step_number % self.steps_copy_target == 0:
            self.copy_weights_to_target_dqn()

        self.optimiser.zero_grad()
        tensor_current_states, tensor_actions, tensor_rewards, tensor_next_states, buffer_indices = transitions
        network_predictions = self.q_network.forward(tensor_current_states)

        tensor_predicted_q_value_current_state = torch.gather(network_predictions, 1, tensor_actions.long())

        # This uses the target network to predict, the qnetwork to generate the greedy action, double Q
        tensor_next_states_values = self.return_next_state_values_tensor(tensor_next_states)
        tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_next_states_values

        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_predicted_q_value_current_state)
        loss.backward()

        self.optimiser.step()

        # Update the TD errors of the transitions we used
        td_error = np.abs((tensor_bellman_current_state_value - tensor_predicted_q_value_current_state).detach().numpy()).ravel()
        for index, error in zip(buffer_indices, td_error):
            self.replay_buffer.transition_td_errors[index] = error

        print("td", step_number, np.mean(td_error))
        print(self.episode_counter)

        # increase epsilon later as we go through episodes and hopefully know more about the initial areas
        if step_number % self.episode_length == 0:
            self.episode_counter += 1

            if self.is_epsilon_greedy:
                self.greedy_counter += 1
                self.epsilon = self.start_epsilon_greedy
                self.steps_increase_epsilon += 2

            if self.greedy_counter == 3:
                self.is_epsilon_delta = True
                self.epsilon = self.start_epsilon_delta
                self.is_epsilon_greedy = False
                self.greedy_counter = 0

        step_in_episode = step_number % self.episode_length

        # Do not do any of this if we are still in random exploration phase
        if step_number > self.steps_made_in_exploration:
        # Linear Epsilon Delta Decrease
            if self.is_epsilon_delta:
                self.epsilon -= self.epsilon_decrease
                if self.epsilon <= 0.3:
                    self.is_epsilon_delta = False
                    self.is_epsilon_greedy = True

            # Linear Epsilon Delta Increase
            elif self.is_epsilon_greedy and step_in_episode > self.steps_increase_epsilon:
                self.epsilon += self.epsilon_increase
                if self.episode_length - step_in_episode < 40:
                    self.epsilon += 0.05
                if self.episode_length - step_in_episode < 10:
                    self.epsilon = 0.2

        self.epsilon = min(1, self.epsilon)
        self.epsilon = max(0, self.epsilon)

        # # LEARNING RATE UPDATE TODO with starting rate at 0.003
        # if self.episode_counter == 10:
        #     self.optimiser.param_groups[0]["lr"] = 0.0001
        # elif self.episode_counter == 15:
        #     self.optimiser.param_groups[0]["lr"] = 0.0005
        # elif self.episode_counter == 20:
        #     self.optimiser.param_groups[0]["lr"] = 0.0003

        return loss.item()

    def return_greedy_action(self, current_state):
        input_tensor = torch.tensor(current_state).float().unsqueeze(0)
        network_prediction = self.q_network.forward(input_tensor)
        return int(network_prediction.argmax())

    def return_next_state_values_tensor(self, tensor_next_states):
        # Double Q
        tensor_network_predictions = self.q_network.forward(tensor_next_states)
        tensor_greedy_actions = tensor_network_predictions.argmax(axis=1).reshape(-1,1)

        with torch.no_grad():
            tensor_target_network_predictions = self.target_q_network.forward(tensor_next_states)

        tensor_next_state_values = torch.gather(tensor_target_network_predictions, 1, tensor_greedy_actions)
        return tensor_next_state_values

    def copy_weights_to_target_dqn(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def epsilon_greedy_policy(self, greedy_action):
        print("EPS", self.epsilon)
        if np.random.randint(0, 100) in range(int(self.epsilon * 100)):
            return np.random.randint(0, 8), False
        else:
            return greedy_action, True

class ReplayBuffer:
    def __init__(self, max_capacity, batch_size=50):
        self.buffer_max_len = max_capacity
        self.replay_buffer = collections.deque(maxlen=self.buffer_max_len)
        self.batch_size = batch_size
        self.length = 0
        # Weights are calculated from the TD errors
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
    def generate_batch(self, batch_size = False):
        # REMOVE THIS IF NOT NEEDED, IE IF CONSTANT BATCH SIZE # TODO
        if not batch_size:
            batch_size = self.batch_size

        # Adding a min probability constant to make sure transitions with small errors are still selected
        min_probability_constant = 0.5 # TODO

        # Normalise weights
        weights = (np.array(self.transition_td_errors) + min_probability_constant) / (
                    sum(self.transition_td_errors) + min_probability_constant * self.length)

        current_states = []
        actions = []
        rewards = []
        next_states = []

        # We generate random indices according to their TD error weights
        indices = np.random.choice(range(self.length), batch_size, replace=False, p=weights)

        # We add the last transition to the buffer so it is trained on for sure, from this we will then get the TD error
        # We replace the last transition picked, this will likely have the lowest prob and be least important, we do
        # this because append is slow and creates a copy
        indices[-1] = self.length - 1
        for index in indices:
            current_states.append(self.replay_buffer[index][0])  # 1x2
            actions.append([self.replay_buffer[index][1]])  # 1x1
            rewards.append([self.replay_buffer[index][2]])  # 1x1
            next_states.append(self.replay_buffer[index][3])  # 1x2
        return torch.tensor(current_states).float(), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(
            next_states).float(), indices  # MSE needs float values, so cast rewards to floats

