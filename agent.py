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
    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=200)
        self.layer_2 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_3 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_4 = torch.nn.Linear(in_features=200, out_features=200)
        self.layer_5 = torch.nn.Linear(in_features=200, out_features=100)
        self.layer_6 = torch.nn.Linear(in_features=200, out_features=100)
        self.layer_7 = torch.nn.Linear(in_features=200, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=300, out_features=output_dimension)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.layer_5.weight)
        torch.nn.init.xavier_uniform_(self.layer_6.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input):
        layer_1_output = torch.nn.functional.leaky_relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_1_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_1_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_2_output))
        layer_6_output = torch.nn.functional.relu(self.layer_6(layer_3_output))
        layer_7_output = torch.nn.functional.relu(self.layer_7(layer_4_output))
        layer_concat = torch.cat((layer_7_output, layer_5_output, layer_6_output), dim=1)
        output = self.output_layer(layer_concat)
        return output


class Agent:
    def __init__(self):
        # Replay buffer batch size
        self.batch_size = 50
        self.episode_length = 200 # 250 TODO
        self.actual_episode_length = self.episode_length
        self.episode_counter = 0

        # Set random exploration episode length
        self.random_exploration_episode_length = 10000 # MAKE SHORTER so less imbalance? add one full random again?
        self.stop_exploration = False
        self.steps_exploration_episode_cutoff = 300
        self.initial_area_exploration = True
        self.done_initial_area_exploration = False
        self.reached_goal = False
        self.got_stuck = False

        self.start_training = False
        self.first_train = True

        self.num_steps_taken = 0
        self.state = None
        self.action = None

        # Replay buffer
        self.buffer_size = 4000000 # no effect when have a good solution and keep going back to goal, make this smaller
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        # DQN
        self.dqn = DQN(self.batch_size, replay_buffer_size=self.buffer_size)
        self.dqn.copy_weights_to_target_dqn()
        self.dqn.episode_length = self.episode_length
        self.dqn.steps_copy_target = self.episode_length
        self.dqn.replay_buffer = self.replay_buffer

        # 8 actions
        self.actions = np.array([[-0.02, 0], [-0.01414, -0.01414],
                                 [0, -0.02], [0.01414, -0.01414],
                                 [0.02, 0],  [0.01414, 0.01414],
                                 [0, 0.02],  [-0.01414, 0.01414]])

    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episode_counter += 1
            self.steps_taken_in_episode = 0
            self.got_stuck = False

            if self.reached_goal:
                self.stop_exploration = True
                self.initial_area_exploration = True
                self.num_steps_taken = 0

            if self.done_initial_area_exploration:
                self.start_training = True
                self.initial_area_exploration = False
                self.num_steps_taken = 0

            # Decrease episode length every time we reach the goal
            if self.dqn.has_reached_goal_previous_episode:
                self.episode_length -= 5
                self.episode_length = max(100, self.episode_length)
                # self.actual_episode_length -= 5 # TODO
                # self.actual_episode_length = max(100, self.actual_episode_length)
                print("DECREASING EP LENGTH", self.episode_length)
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state: np.ndarray):  # TODO REMOVE IS GREEDY FROM RETURN
        # RANDOM EXPLORATION IN BEGINNING
        is_greedy = False  # TODO REMOVE IS GREEDY FROM RETURN
        # Random exploration in beginning, try every direction
        if not self.stop_exploration:
            # If we cannot find a quick way out of the starting area we need to explore fully randomly
            if self.episode_counter > 50:
                action = np.random.randint(8)
                self.steps_exploration_episode_cutoff = 200
            else:
                self.episode_length = self.random_exploration_episode_length
                direction = (self.episode_counter - 1) % 8
                # If get stuck early in the episode, restart with the next action
                if self.steps_taken_in_episode < 9 and self.got_stuck:
                    # Break episode early
                    self.episode_length = self.num_steps_taken + 1
                    action = 0
                elif self.steps_taken_in_episode < 25 and not self.got_stuck:
                    action = direction
                else:
                    action = np.random.randint(8)

        # If we have reached the goal once, explore a bit more of the starting area before we start training  #TODO
        elif self.initial_area_exploration:
            print("initial AREA") # TODO

            self.episode_length = 600
            action = np.random.randint(8)
            self.done_initial_area_exploration = True

        # No more exploration
        else:
            self.episode_length = self.actual_episode_length
            greedy_action = self.dqn.return_greedy_action(state)
            action, is_greedy = self.dqn.epsilon_greedy_policy(greedy_action)  # TODO REMOVE IS GREEDY FROM RETURN

        self.steps_taken_in_episode += 1
        self.num_steps_taken += 1
        self.state = state
        # Store action as an int
        self.action = action
        action = np.array(self.actions[action])
        return action, is_greedy  # TODO REMOVE IS GREEDY FROM RETURN

    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if not self.stop_exploration:
            # If we are not close enough to the goal, we will restart exploration
            if distance_to_goal > 0.8 and self.steps_taken_in_episode > self.steps_exploration_episode_cutoff:
                # End the episode
                self.episode_length = self.num_steps_taken

                # Clean the accumulated buffers
                self.replay_buffer.clear()
                self.replay_buffer.distance_errors.clear()
                self.replay_buffer.length = 0
                self.replay_buffer.wrap_around_index = 0

            # If we are close enough to the goal, we will start training after a final set of exploration
            if distance_to_goal < 0.008:
                self.reached_goal = True
                self.replay_buffer.convert_deque_to_array()

        if np.linalg.norm(self.state - next_state) < 0.0002:
            self.got_stuck = True
        else:
            self.got_stuck = False

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
        elif distance_to_goal < 0.6: # TODO ADD MORE REWARDS FOR MORE DIFFICULT LEVELS?
            reward = 0.02
        else:
            reward = 0

        # TODO
        if reward > 0:
            print("reward", reward)

        # Store the distances for the buffer to use, not used in agent
        distance_rounded = round(distance_to_goal, 2)
        if self.stop_exploration:
            # array_index = self.replay_buffer.wrap_around_index % self.replay_buffer.buffer_max_len
            # NEED TO OFFSET WHEN INDEXING DEQUE AS WELL can do with constant offset TODO
            self.replay_buffer.distance_errors_array[self.replay_buffer.length] = distance_rounded
        else:
            self.replay_buffer.distance_errors.append(distance_rounded)

        # If the current distance is the largest or smallest, record that for use in buffer batching
        if distance_rounded > self.replay_buffer.max_distance:
            self.replay_buffer.max_distance = distance_rounded
        if distance_rounded < self.replay_buffer.min_distance:
            self.replay_buffer.min_distance = distance_rounded

        # Do this after so the length doesnt change before the above
        transition = (self.state, self.action, reward, next_state)
        self.replay_buffer.add(transition)

        if self.start_training:
            # For the first time training, repeat training for 2000 iterations
            if self.first_train:
                for i in range(2000):
                    print("first train", i)
                    # TODO try bigger batch with multiple selections and replace?
                    self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size),
                                                   self.num_steps_taken, distance_to_goal)
                self.first_train = False

            self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size), self.num_steps_taken,
                                           distance_to_goal)

    def get_greedy_action(self, state: np.ndarray):
        return self.actions[self.dqn.return_greedy_action(state)]


class DQN:
    gamma = .95

    # The class initialisation function.
    def __init__(self, batch_size, replay_buffer_size):
        self.q_network = Network(input_dimension=2, output_dimension=8)
        self.target_q_network = Network(input_dimension=2, output_dimension=8)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Episode length
        self.episode_length = None
        self.episode_counter = 0
        self.steps_copy_target = self.episode_length

        # Epsilon
        self.epsilon = 0.5  # TODO
        self.start_epsilon_delta = 0.55
        self.end_epsilon_delta = 0.3
        self.epsilon_decrease = 0.00003

        self.has_reached_goal_previous_episode = False

    def epsilon_greedy_policy(self, greedy_action):
        print(self.epsilon)
        if np.random.randint(0, 100) in range(int(self.epsilon * 100)):
            random_action = np.random.randint(0, 8)
            # while random_action == greedy_action:
            #     random_action = np.random.randint(0, 8)
            return random_action, False
        else:
            return greedy_action, True

    def train_q_network_batch(self, transitions: tuple, step_number, distance_to_goal):
        print(self.episode_counter) # TODO
        # Update target network
        if step_number % self.steps_copy_target == 0:
            self.copy_weights_to_target_dqn()

        tensor_current_states, tensor_actions, tensor_rewards, tensor_next_states = transitions

        self.optimiser.zero_grad()

        # Current state values
        network_predictions = self.q_network.forward(tensor_current_states)
        tensor_predicted_q_value_current_state = torch.gather(network_predictions, 1, tensor_actions.long())

        # Next state values, Double Q using target network to predict values
        tensor_next_states_values = self.return_next_state_values_tensor(tensor_next_states)

        # Bellman equation
        tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_next_states_values

        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_predicted_q_value_current_state)
        loss.backward()

        self.optimiser.step()

        # increase epsilon later as we go through episodes and hopefully know more about the initial areas
        if step_number % self.episode_length == 0:
            self.episode_counter += 1

            if self.has_reached_goal_previous_episode:
                self.start_epsilon_delta -= 0.01
                self.end_epsilon_delta -= 0.01
                self.has_reached_goal_previous_episode = False
                self.start_epsilon_delta = max(self.start_epsilon_delta, 0.3)
                # self.end_epsilon_delta = max(self.end_epsilon_delta, 0.05)

        # Linear Epsilon Delta Decrease
        if self.epsilon > 0.4:
            self.epsilon -= self.epsilon_decrease
        else:
            self.epsilon -= 0.0001

        # If epsilon drops below the end threshold restart
        if self.epsilon < self.end_epsilon_delta:
            self.epsilon = self.start_epsilon_delta

        # If we have reached the goal, decrease the starting epsilon
        if distance_to_goal < 0.03:
            self.has_reached_goal_previous_episode = True
        return loss.item()

    def return_greedy_action(self, current_state):
        input_tensor = torch.tensor(current_state).float().unsqueeze(0)
        network_prediction = self.q_network.forward(input_tensor)
        return int(network_prediction.argmax())

    def return_next_state_values_tensor(self, tensor_next_states):
        # Double Q
        # Use Q network to get greedy actions
        tensor_network_predictions = self.q_network.forward(tensor_next_states)
        tensor_greedy_actions = tensor_network_predictions.argmax(axis=1).reshape(-1, 1)

        # Use target network to predict state values
        with torch.no_grad():
            tensor_target_network_predictions = self.target_q_network.forward(tensor_next_states)
        tensor_next_state_values = torch.gather(tensor_target_network_predictions, 1, tensor_greedy_actions)
        return tensor_next_state_values

    def copy_weights_to_target_dqn(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class ReplayBuffer:
    def __init__(self, max_capacity, batch_size=50):
        self.buffer_max_len = max_capacity
        self.replay_buffer = collections.deque(maxlen=self.buffer_max_len)
        self.batch_size = batch_size
        self.length = 0
        self.wrap_around_index = 0
        self.index_offset = 0

        self.max_distance = 0
        self.min_distance = 2

        # Weights are calculated from the TD errors
        self.distance_errors = collections.deque(maxlen=self.buffer_max_len) # THIS NEEDS TO BE NP ARRAY INSTEAD after reached goal convert to np array
        self.distance_errors_array = np.empty(self.buffer_max_len)

        self.indices = np.zeros(self.batch_size + 1).astype(int)

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)
        # Adds 1 only if the buffer is not full
        self.length += 1 if self.length < self.buffer_max_len else 0
        self.wrap_around_index += 1
        self.index_offset = self.wrap_around_index % self.buffer_max_len

    def clear(self):
        self.replay_buffer.clear()

    def convert_deque_to_array(self): # THIS NEEDS TO BE CALLED IN THE ENDING CONDITION
        for i in range(self.length):
            self.distance_errors_array[i] = self.distance_errors[i]

    # Returns tuple of tensors, each has dimension (batch_size, *), SARS'
    def generate_batch(self, batch_size):
        # Distance weights
        indices = []
        for distance in np.round(np.linspace(self.min_distance, self.max_distance, num=batch_size, endpoint=True), decimals=2):
            samples_at_distance = np.argwhere(self.distance_errors_array[:self.length] == distance).ravel()
            while len(samples_at_distance) == 0:
                distance = round(distance - 0.01, 2)
                samples_at_distance = np.argwhere(self.distance_errors_array == distance).ravel()
            # try: # DOUBLE BATCH
            #     indices.extend(np.random.choice(samples_at_distance, 2, replace=False))
            # except:
            #     print("ONLY ONE SAMPLE AVAILABLE")
            #     indices.append(np.random.choice(samples_at_distance))
            indices.append(np.random.choice(samples_at_distance))

        # We add the last transition to the buffer so it is trained on for sure
        indices.append(self.length - 1) # NEEDS OFFSET WHEN NP ARRAY

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
