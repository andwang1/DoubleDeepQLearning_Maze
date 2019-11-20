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

import time  # TODO


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=200)  # OVERFITTING?
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

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
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
    # Function to initialise the agent
    def __init__(self):
        # Replay buffer batch size
        self.batch_size = 50
        self.episode_length = 500 # Need long to reach goal early
        self.actual_episode_length = self.episode_length
        self.episode_counter = 0

        # Set random exploration episode length
        self.random_exploration_episode_length = 10000 # MAKE SHORTER so less imbalance? add one full random again?
        self.steps_made_in_exploration = self.random_exploration_episode_length * 6
        self.stop_exploration = False
        self.steps_exploration_episode_cutoff = 300

        # self.training_threshhold = self.random_exploration_episode_length * 8
        self.training_threshhold = 15900 # CUTOFF TIMES 3
        self.start_training = False
        self.first_train = True

        self.num_steps_taken = 0
        self.state = None
        self.action = None

        # Replay buffer
        # self.buffer_size = self.steps_made_in_exploration + self.random_exploration_episode_length
        self.buffer_size = 400000 # no effect when have a good solution and keep going back to goal, make this smaller
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

            if self.stop_exploration:
                self.start_training = True

            # Decrease episode length every time we reach the goal
            if self.dqn.has_reached_goal_previous_episode:
                print("DECREASING EP LENGTH")
                self.episode_length -= 5
                self.episode_length = max(100, self.episode_length)
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state: np.ndarray):  # TODO REMOVE IS GREEDY FROM RETURN
        # RANDOM EXPLORATION IN BEGINNING
        is_greedy = False  # TODO REMOVE IS GREEDY FROM RETURN
        # while we are in the number of steps try every direction and if get stuck in less than 8 end episode
        if not self.stop_exploration:
            self.episode_length = self.random_exploration_episode_length
            direction = (self.episode_counter - 1) % 8
            # IF GET STUCK EARLY THEN WE ARE NEXT TO WALL, quit this random exploration early

            if self.steps_taken_in_episode < 9 and self.got_stuck:
                # Break episode early
                self.episode_length = self.num_steps_taken + 1
                action = 0

            elif self.steps_taken_in_episode < 25 and not self.got_stuck:
                action = direction
            else:
                action = np.random.randint(8)

        else:
            self.episode_length = self.actual_episode_length
            action, is_greedy = self.dqn.epsilon_greedy_policy(
                self.dqn.return_greedy_action(state))  # TODO REMOVE IS GREEDY FROM RETURN

        self.steps_taken_in_episode += 1
        self.num_steps_taken += 1
        self.state = state  # NP ARRAY
        self.action = action
        action = np.array(self.actions[action])
        return action, is_greedy  # return here as nparray # TODO REMOVE IS GREEDY FROM RETURN

    # AFTER ACTION CALL THIS GETS CALLED GET THE TRANSITION HERE TODO
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if not self.stop_exploration:
            # BREAK CONDITION FOR EXPLORATION if have not gotten far enough
            if distance_to_goal > 0.8 and self.steps_taken_in_episode > self.steps_exploration_episode_cutoff:
                self.episode_length = self.num_steps_taken
                self.replay_buffer.clear()
                self.replay_buffer.transition_td_errors.clear()
                self.replay_buffer.distance_errors.clear()
                self.replay_buffer.length = 0
                self.steps_taken_in_episode = 0 # TO SKIP TRAINING
                print("ENDING EARLY", self.steps_taken_in_episode)

            # Break condition if have gotten close enough
            if distance_to_goal < 0.02:
                self.stop_exploration = True
                self.replay_buffer.convert_deque_to_array()
                # Break the episode
                # self.episode_length = self.num_steps_taken + 20

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
        elif distance_to_goal < 0.6:
            reward = 0.02
        elif distance_to_goal < 0.7:
            reward = 0.01
        elif distance_to_goal < 0.8:
            reward = 0.005
        else:
            reward = 0
        # reward = 1 - distance_to_goal
        if reward > 0:
            print("reward", reward)


        # Add new weight of 1 for the newest transition, we will make sure this gets picked manually by adding to batch
        # Cannot make this 0 for some reason will give error ValueError: probabilities contain NaN
        # self.replay_buffer.transition_td_errors.append(0.0001)
        distance_rounded = round(distance_to_goal, 2)
        if not self.stop_exploration:
            self.replay_buffer.distance_errors.append(distance_rounded)
        else:
            self.replay_buffer.distance_errors_array[self.replay_buffer.length] = distance_rounded

        if distance_rounded > self.replay_buffer.max_distance:
            self.replay_buffer.max_distance = distance_rounded
        if distance_rounded < self.replay_buffer.min_distance:
            self.replay_buffer.min_distance = distance_rounded

        # Do this after so the length doesnt change before the above
        transition = (self.state, self.action, reward, next_state)
        self.replay_buffer.add(transition)


        # Train
        # if self.num_steps_taken > self.training_threshhold and self.steps_taken_in_episode > self.batch_size:
        if self.start_training:
            if self.first_train:
                for _ in range(1000):
                    print("enters first train loop")
                    self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size),
                                                   self.num_steps_taken,
                                                   self.got_stuck, distance_to_goal)
                self.first_train = False
                # raise

            self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size), self.num_steps_taken,
                                           self.got_stuck, distance_to_goal)

    def get_greedy_action(self, state: np.ndarray):
        return self.actions[self.dqn.return_greedy_action(state)]


class DQN:
    gamma = .95

    # The class initialisation function.
    def __init__(self, batch_size, replay_buffer_size):
        self.q_network = Network(input_dimension=2, output_dimension=8)
        self.target_q_network = Network(input_dimension=2, output_dimension=8)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Batch size used in the replay_buffer
        self.replay_buffer = None
        self.batch_size = batch_size  # TODO update if change the batch size in replay buffer
        self.replay_buffer_size = replay_buffer_size  # TODO FEED IN REPLAY BUFFER DIRECTLY

        # Episode length
        self.episode_length = None
        self.episode_counter = 0
        self.steps_copy_target = self.episode_length

        # Epsilon
        self.epsilon = 0.5  # TODO
        self.steps_increase_epsilon = 15
        self.saved_epsilon = self.epsilon

        # is greedy
        self.is_greedy = False
        self.epsilon_maxed = False
        self.used_saved_epsilon = False

        # # Epsilon linear in episode length
        self.is_epsilon_delta = True
        self.is_epsilon_greedy = False

        self.start_epsilon_delta = 0.55
        self.start_epsilon_greedy = 0.3

        self.epsilon_decrease = 0.00003 # MAKE LOWER
        self.epsilon_increase = 0.001

        self.steps_made_in_exploration = 0
        self.greedy_counter = 0

        self.has_reached_goal_previous_episode = False

        self.all_actions = set(range(8))


    def epsilon_greedy_policy(self, greedy_action):
        # If we are in end exploration mode in epsilon greedy part
        if self.epsilon <= -90:
            time.sleep(.5)
            print("enter loop")
            print("greedy action", greedy_action)
            if greedy_action == 2: # DOWN
                likely_next_actions = [3, 4, 5, 6]
                if np.random.randint(0, 100) in range(90):
                    return np.random.choice(likely_next_actions), False
                else:
                    return np.random.choice(list(self.all_actions - set(likely_next_actions) - {greedy_action})), False

            if greedy_action == 3: # DIAG DOWN RIGHT
                likely_next_actions = [6, 5]
                if np.random.randint(0, 100) in range(90):
                    return np.random.choice(likely_next_actions), False
                else:
                    return np.random.choice(list(self.all_actions - set(likely_next_actions) - {greedy_action})), False

            if greedy_action == 4: # RIGHT
                likely_next_actions = [6, 5, 4, 3]
                if np.random.randint(0, 100) in range(90):
                    return np.random.choice(likely_next_actions), False
                else:
                    return np.random.choice(list(self.all_actions - set(likely_next_actions) - {greedy_action})), False

            if greedy_action == 5: # DIAG UP
                likely_next_actions = [2, 3]
                if np.random.randint(0, 100) in range(90):
                    if np.random.randint(0, 100) in range(70):
                        return 2, False
                    else:
                        return 3, False
                else:
                    return np.random.choice(list(self.all_actions - set(likely_next_actions) - {greedy_action})), False

            if greedy_action == 6: # UP
                likely_next_actions = [2, 3, 4, 5]
                if np.random.randint(0, 100) in range(80):
                    return np.random.choice(likely_next_actions), False
                else:
                    return np.random.choice(list(self.all_actions - set(likely_next_actions) - {greedy_action})), False

            # If its optimal to go any other way TODO? WRITE OUT ALL ACTIONS IF IT WORKS
            return np.random.randint(8), False

        else:
            # Standard epsilon greedy
            print("EXEC EPS GREEDY", self.epsilon)
            if np.random.randint(0, 100) in range(int(self.epsilon * 100)):
                random_action = np.random.randint(0, 8)
                # while random_action == greedy_action:
                #     random_action = np.random.randint(0, 8)
                return random_action, False
            else:
                return greedy_action, True

    def train_q_network_batch(self, transitions: tuple, step_number, got_stuck, distance_to_goal):
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
        # td_error = np.abs(
        #     (tensor_bellman_current_state_value - tensor_predicted_q_value_current_state).detach().numpy()).ravel()
        # for index, error in zip(buffer_indices, td_error):
        #     self.replay_buffer.transition_td_errors[index] = error
        #
        # print("td", step_number, np.mean(td_error))
        print(self.episode_counter)

        # increase epsilon later as we go through episodes and hopefully know more about the initial areas
        if step_number % self.episode_length == 0:
            self.episode_counter += 1

            if self.has_reached_goal_previous_episode:
                self.start_epsilon_delta -= 0.01
                self.has_reached_goal_previous_episode = False
                self.start_epsilon_delta = max(self.start_epsilon_delta, 0.3)

            if self.epsilon <= 0.2:
                self.epsilon = self.start_epsilon_delta
            elif self.used_saved_epsilon is not False:
                self.epsilon = self.saved_epsilon

            self.saved_epsilon = False
            self.used_saved_epsilon = False

        step_in_episode = step_number % self.episode_length

        # Do not do any of this if we are still in random exploration phase
        # Linear Epsilon Delta Decrease
        if self.epsilon > 0.4:
            self.epsilon -= self.epsilon_decrease
        else:
            self.epsilon -= 0.0001
            if self.episode_length - step_in_episode < self.episode_length / 3 and distance_to_goal > 0.3:
                if self.saved_epsilon is False:
                    self.saved_epsilon = self.epsilon
                self.used_saved_epsilon = True
                self.epsilon = 0.5


        # self.epsilon = max(0, self.epsilon)

        if distance_to_goal < 0.03:
            self.has_reached_goal_previous_episode = True

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
        tensor_greedy_actions = tensor_network_predictions.argmax(axis=1).reshape(-1, 1)

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

        self.max_distance = 0
        self.min_distance = 2

        # Weights are calculated from the TD errors
        self.transition_td_errors = collections.deque(maxlen=self.buffer_max_len)
        self.distance_errors = collections.deque(maxlen=self.buffer_max_len) # THIS NEEDS TO BE NP ARRAY INSTEAD after reached goal convert to np array
        self.distance_errors_array = np.empty(self.buffer_max_len)


    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)
        # Adds 1 only if the buffer is not full
        self.length += 1 if self.length < self.buffer_max_len else 0

    def clear(self):
        self.replay_buffer.clear()

    def convert_deque_to_array(self): # THIS NEEDS TO BE CALLED IN THE ENDING CONDITION
        for i in range(self.length):
            self.distance_errors_array[i] = self.distance_errors[i]

    # Returns tuple of tensors, each has dimension (batch_size, *), SARS'
    def generate_batch(self, batch_size):
        # Adding a min probability constant to make sure transitions with small errors are still selected
        min_probability_constant = 0.5  # TODO

        # Distance weights
        # Calculate weights by iterating through array
        # print("before indices")
        indices = []
        # print(np.linspace(self.min_distance, self.max_distance, num=batch_size, endpoint=True))
        # print(np.round(np.linspace(self.min_distance, self.max_distance, num=batch_size, endpoint=True), decimals=2))
        for distance in np.round(np.linspace(self.min_distance, self.max_distance, num=batch_size, endpoint=True), decimals=2):
            # print(self.length)
            samples_at_distance = np.argwhere(self.distance_errors_array[:self.length] == distance).ravel()
            # print(samples_at_distance)
            while len(samples_at_distance) == 0:
                # print("whileloop")
                distance = round(distance - 0.01, 2)
                # print(distance)
                samples_at_distance = np.argwhere(self.distance_errors_array == distance).ravel()
                # print(samples_at_distance)

            indices.append(np.random.choice(samples_at_distance))

        # print("calculated indices")
        # Normalise weights
        # weights = (np.array(self.transition_td_errors) + min_probability_constant) / (
        #         sum(self.transition_td_errors) + min_probability_constant * self.length)

        current_states = []
        actions = []
        rewards = []
        next_states = []

        # We generate random indices according to their TD error weights
        # indices = np.random.choice(range(self.length), batch_size, replace=False, p=weights)

        # We add the last transition to the buffer so it is trained on for sure, from this we will then get the TD error
        # We replace the last transition picked, this will likely have the lowest prob and be least important, we do
        # this because append is slow and creates a copy
        # indices[-1] = self.length - 1

        # print(indices)
        # print(self.length)
        # print(len(self.replay_buffer))
        # print(len(self.distance_errors_array))

        for index in indices:
            current_states.append(self.replay_buffer[index][0])  # 1x2
            actions.append([self.replay_buffer[index][1]])  # 1x1
            rewards.append([self.replay_buffer[index][2]])  # 1x1
            next_states.append(self.replay_buffer[index][3])  # 1x2

        return torch.tensor(current_states).float(), torch.tensor(actions), \
               torch.tensor(rewards).float(), torch.tensor(next_states).float(), indices
