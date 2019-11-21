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

import numpy as np
import torch
import collections


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()

        # 3 not fully connected layers
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=200)
        self.layer_2 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_3 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_4 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_5 = torch.nn.Linear(in_features=200, out_features=100)
        self.layer_6 = torch.nn.Linear(in_features=200, out_features=100)
        self.layer_7 = torch.nn.Linear(in_features=200, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=300, out_features=output_dimension)

        # Initialise weights
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.layer_5.weight)
        torch.nn.init.xavier_uniform_(self.layer_6.weight)
        torch.nn.init.xavier_uniform_(self.layer_7.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input):
        # Forward through layers and concatenate for final output layer
        layer_1_output = torch.nn.functional.leaky_relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_1_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_1_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_2_output))
        layer_6_output = torch.nn.functional.relu(self.layer_6(layer_3_output))
        layer_7_output = torch.nn.functional.relu(self.layer_7(layer_4_output))
        layer_concat = torch.cat((layer_5_output, layer_6_output, layer_7_output), dim=1)
        output = self.output_layer(layer_concat)
        return output


class Agent:
    def __init__(self):
        # Episode details
        self.episode_counter = 0
        self.episode_length = 150  # test on map 2 150 with decreasing ep length TODO TEST
        self.actual_episode_length = self.episode_length

        # We randomly explore in alternating directions until we find the goal
        self.random_exploration_episode_length = 8000  # TODO TEST WITH 10k again
        self.distance_to_goal_threshold = 0.008
        self.reached_goal = False
        self.stop_exploration = False

        # If after some steps we are still a certain distance away from goal, restart the exploration
        self.steps_exploration_episode_cutoff = 300
        self.exploration_min_distance = 0.8

        # If we cannot find a way out of the initial area, we use fully random actions to find the goal
        self.undirected_random_exploration = False

        # Once we have found the goal, we do one final exploration episode to explore the initial state further
        self.initial_area_exploration = True
        self.done_initial_area_exploration = False

        # After exploration is done, we will start training
        self.train_now = False
        self.first_time_training = True

        self.num_steps_taken = 0
        self.got_stuck = False
        self.state = None
        self.action = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # DQN
        self.dqn = DQN()
        self.dqn.copy_weights_to_target_dqn()
        self.dqn.episode_length = self.episode_length
        self.dqn.num_steps_copy_target = self.episode_length
        self.dqn.replay_buffer = self.replay_buffer

        # 8 discrete actions
        self.actions = np.array([[-0.02, 0], [-0.01414, -0.01414],
                                 [0, -0.02], [0.01414, -0.01414],
                                 [0.02, 0], [0.01414, 0.01414],
                                 [0, 0.02], [-0.01414, 0.01414]])

    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episode_counter += 1
            self.steps_taken_in_episode = 0
            self.got_stuck = False

            # If we reached the goal during random exploration, explore the initial area
            if self.reached_goal:
                self.stop_exploration = True
                self.initial_area_exploration = True
                self.num_steps_taken = 0

            # If we explored the initial area, start training
            if self.done_initial_area_exploration:
                self.train_now = True
                self.initial_area_exploration = False
                self.num_steps_taken = 0

            return True
        else:
            return False

    def get_next_action(self, state):  # TODO REMOVE IS GREEDY FROM RETURN
        is_greedy = False  # TODO REMOVE IS GREEDY FROM RETURN
        # Random exploration
        if not self.stop_exploration:
            # In the beginning we try every direction a few times to see if we can quickly leave the starting area
            if self.episode_counter < 400:
                self.episode_length = self.random_exploration_episode_length
                direction = (self.episode_counter - 1) % 8

                # If get stuck early in the episode, restart with the next action
                if self.steps_taken_in_episode < 13 and self.got_stuck:
                    self.episode_length = self.num_steps_taken + 1
                    action = 4
                elif self.steps_taken_in_episode < 25 and not self.got_stuck:
                    action = direction
                else:
                    action = np.random.randint(8)

            # If we cannot find a quick way out of the starting area we need to explore fully randomly
            else:
                self.undirected_random_exploration = True
                self.exploration_min_distance = 1.1
                self.distance_to_goal_threshold = 0.1  # TODO
                self.episode_length = 15000
                action = np.random.randint(8)

        # If we have reached the goal once, explore a bit more of the starting area before we start training
        elif self.initial_area_exploration:
            self.episode_length = 800
            self.done_initial_area_exploration = True
            action = np.random.randint(8)

        # After exploration we only use epsilon greedy actions
        else:
            self.episode_length = self.actual_episode_length
            greedy_action = self.dqn.return_greedy_action(state)
            action, is_greedy = self.dqn.epsilon_greedy_policy(greedy_action)  # TODO REMOVE IS GREEDY FROM RETURN

        self.steps_taken_in_episode += 1
        self.num_steps_taken += 1
        self.state = state

        # Store action as an int, return as a np.ndarray
        self.action = action
        action = np.array(self.actions[action])
        return action, is_greedy  # TODO REMOVE IS GREEDY FROM RETURN

    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if not self.stop_exploration:
            # If after a certain number of steps we are not close enough to the goal, we will restart exploration
            if distance_to_goal > self.exploration_min_distance and \
                    self.steps_taken_in_episode > self.steps_exploration_episode_cutoff:
                # Clear the buffers
                self.replay_buffer.clear()
                self.replay_buffer.distance_errors.clear()
                self.replay_buffer.length = 0

                # End the episode
                self.episode_length = self.num_steps_taken

            # If we are close enough to the goal, we will start training after a final set of exploration
            if distance_to_goal < self.distance_to_goal_threshold:
                self.reached_goal = True
                # Convert the deque we have been using to a np.ndarray for faster processing
                self.replay_buffer.convert_deque_to_array()

                # If we are already fully random, we have lost a lot of time, and exploration episodes are long,
                # stop immediately when reaching the goal
                if self.undirected_random_exploration:
                    self.episode_length = self.num_steps_taken

        # Record whether agent got stuck on the last move
        if np.linalg.norm(self.state - next_state) < 0.0002:
            self.got_stuck = True
        else:
            self.got_stuck = False

        # Sparse rewards
        if distance_to_goal < 0.03:
            reward = 1
        elif distance_to_goal < 0.05:
            reward = 0.5
        elif distance_to_goal < 0.1:
            reward = 0.1
        elif distance_to_goal < 0.2:
            reward = 0.07
        elif distance_to_goal < 0.3:
            reward = 0.05
        elif distance_to_goal < 0.4:
            reward = 0.04
        elif distance_to_goal < 0.5:
            reward = 0.03
        elif distance_to_goal < 0.6:  # TODO ADD MORE REWARDS FOR MORE DIFFICULT LEVELS?
            reward = 0.02
        else:
            reward = 0

        # Store distances for the buffer to use
        distance_rounded = round(distance_to_goal, 2)
        # If we are no longer exploring, we are using the np.ndarray not the deque
        if self.stop_exploration:
            self.replay_buffer.distance_errors_array[self.replay_buffer.length] = distance_rounded
        else:
            self.replay_buffer.distance_errors.append(distance_rounded)

        # If the current distance is the largest or smallest out of all distances, recalculate the linspace
        if distance_rounded > self.replay_buffer.max_distance:
            self.replay_buffer.max_distance = distance_rounded
            self.replay_buffer.distance_linspace = np.round(np.linspace(self.replay_buffer.min_distance,
                                                                        self.replay_buffer.max_distance,
                                                                        num=self.replay_buffer.batch_size,
                                                                        endpoint=True),
                                                            decimals=2)
        if distance_rounded < self.replay_buffer.min_distance:
            self.replay_buffer.min_distance = distance_rounded
            self.replay_buffer.distance_linspace = np.round(np.linspace(self.replay_buffer.min_distance,
                                                                        self.replay_buffer.max_distance,
                                                                        num=self.replay_buffer.batch_size,
                                                                        endpoint=True),
                                                            decimals=2)

        # Add the transition to the buffer
        transition = (self.state, self.action, reward, next_state)
        self.replay_buffer.add(transition)

        if self.train_now:
            # For the first time training, repeat training for 2000 iterations
            if self.first_time_training:
                for _ in range(2000):  # TODO
                    self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(),
                                                   self.num_steps_taken, distance_to_goal)
                self.first_time_training = False

            self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(), self.num_steps_taken,
                                           distance_to_goal)

    def get_greedy_action(self, state: np.ndarray):
        return self.actions[self.dqn.return_greedy_action(state)]


class DQN:
    gamma = .95

    # The class initialisation function.
    def __init__(self):
        self.q_network = Network(input_dimension=2, output_dimension=8)
        self.target_q_network = Network(input_dimension=2, output_dimension=8)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Episode length
        self.episode_length = None
        self.episode_counter = 0
        self.num_steps_copy_target = self.episode_length

        # Epsilon
        self.epsilon = 0.5  # TODO
        self.start_epsilon_delta = 0.55
        self.end_epsilon_delta = 0.3

        self.has_reached_goal_previous_episode = False

    def epsilon_greedy_policy(self, greedy_action):
        if np.random.randint(0, 100) in range(int(self.epsilon * 100)):
            random_action = np.random.randint(0, 8)
            return random_action, False
        else:
            return greedy_action, True

    def train_q_network_batch(self, transitions: tuple, step_number, distance_to_goal):
        tensor_current_states, tensor_actions, tensor_rewards, tensor_next_states = transitions

        # Update target network
        if step_number % self.num_steps_copy_target == 0:
            self.copy_weights_to_target_dqn()

        self.optimiser.zero_grad()

        # Current state values
        network_predictions = self.q_network.forward(tensor_current_states)
        tensor_predicted_q_value_current_state = torch.gather(network_predictions, 1, tensor_actions.long())

        # Next state values
        # Double Q, use Q network to get greedy actions, target network to get the value of the next state
        tensor_network_predictions = self.q_network.forward(tensor_next_states)
        tensor_greedy_actions = tensor_network_predictions.argmax(axis=1).reshape(-1, 1)
        # Detach the gradient from the target network tensor
        with torch.no_grad():
            tensor_target_network_predictions = self.target_q_network.forward(tensor_next_states)
        tensor_next_state_values = torch.gather(tensor_target_network_predictions, 1, tensor_greedy_actions)

        # Bellman equation
        tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_next_state_values

        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_predicted_q_value_current_state)
        loss.backward()

        self.optimiser.step()

        # Episode restart, set values for epsilon decay
        if step_number % self.episode_length == 0:
            self.episode_counter += 1
            print(self.episode_counter)  # TODO

            # If we reached the goal the previous episode, we start and end at lower epsilons, down to a threshold
            if self.has_reached_goal_previous_episode:
                self.start_epsilon_delta -= 0.01
                self.end_epsilon_delta -= 0.01
                self.start_epsilon_delta = max(self.start_epsilon_delta, 0.2)
                # self.end_epsilon_delta = max(self.end_epsilon_delta, 0.02) # TODO REVIEW
                self.has_reached_goal_previous_episode = False

        # Linear Epsilon Delta Decrease
        if self.epsilon > 0.4:
            self.epsilon -= 0.00003
        else:
            self.epsilon -= 0.0001

        # If epsilon drops below the end threshold restart
        if self.epsilon < self.end_epsilon_delta:
            self.epsilon = self.start_epsilon_delta

        if distance_to_goal < 0.03:
            self.has_reached_goal_previous_episode = True

        return loss.item()

    def return_greedy_action(self, current_state):
        input_tensor = torch.tensor(current_state).float().unsqueeze(0)
        network_prediction = self.q_network.forward(input_tensor)
        return int(network_prediction.argmax())

    def copy_weights_to_target_dqn(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class ReplayBuffer:
    def __init__(self, max_capacity=4000000, batch_size=50):
        self.buffer_max_len = max_capacity
        self.batch_size = batch_size
        self.replay_buffer = collections.deque(maxlen=self.buffer_max_len)
        self.length = 0

        # Max and min distances from goal
        self.max_distance = 0
        self.min_distance = 2

        # Use distances to sample transitions
        self.distance_errors = collections.deque(maxlen=self.buffer_max_len)
        self.distance_errors_array = np.empty(self.buffer_max_len)
        self.distance_linspace = None

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)
        self.length += 1 if self.length < self.buffer_max_len else 0

    def clear(self):
        self.replay_buffer.clear()

    def convert_deque_to_array(self):
        for i in range(self.length):
            self.distance_errors_array[i] = self.distance_errors[i]

    def generate_batch(self):
        indices = []
        for distance in self.distance_linspace:
            samples_at_distance = np.argwhere(self.distance_errors_array[:self.length] == distance).ravel()

            while len(samples_at_distance) == 0:
                distance = round(distance - 0.01, 2)
                samples_at_distance = np.argwhere(self.distance_errors_array[:self.length] == distance).ravel()

            indices.append(np.random.choice(samples_at_distance))

        # We add the last transition to the buffer so it is trained on for sure
        indices.append(self.length - 1)

        current_states = []
        actions = []
        rewards = []
        next_states = []

        for index in indices:
            current_states.append(self.replay_buffer[index][0])  # 1x2
            actions.append([self.replay_buffer[index][1]])  # 1x1
            rewards.append([self.replay_buffer[index][2]])  # 1x1
            next_states.append(self.replay_buffer[index][3])  # 1x2

        return torch.tensor(current_states).float(), torch.tensor(actions), \
               torch.tensor(rewards).float(), torch.tensor(next_states).float()
