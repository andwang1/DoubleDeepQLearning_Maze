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
        # self.layer_4 = torch.nn.Linear(in_features=200, out_features=200)
        self.output_layer = torch.nn.Linear(in_features=200, out_features=output_dimension)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        # torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.leaky_relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.leaky_relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.leaky_relu(self.layer_3(layer_2_output))
        # layer_4_output = torch.nn.functional.leaky_relu(self.layer_4(layer_3_output))
        output = self.output_layer(layer_3_output)
        return output


class Agent:
    # Function to initialise the agent
    def __init__(self):
        # Replay buffer batch size
        self.batch_size = 40
        # Set the episode length (you will need to increase this)
        self.episode_length = 200 # 100 is the episode they will run at TODO SCALE WITH TIME?
        self.actual_episode_length = self.episode_length
        self.episode_counter = 0

        # Set random exploration episode length
        self.random_exploration_episode_length = 300 #TODO 120, CHANGE FOR TESTING
        self.exploration_length = 8
        self.random_exploration_step_size = 0.015
        self.steps_made_in_exploration = self.random_exploration_episode_length * self.exploration_length

        # Set number of steps at which to start training
        steps_needed_with_batch_to_train = self.steps_made_in_exploration / self.batch_size
        # every sample can be trained on twice
        self.training_threshhold = int(self.steps_made_in_exploration - steps_needed_with_batch_to_train * 2)
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Replay buffer
        self.buffer_size = self.steps_made_in_exploration + self.random_exploration_episode_length
        # self.buffer_size = 400000
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        # Step size for each step
        self.step_length = 0.02  # TODO size of normalisation
        # DQN
        self.dqn = DQN(self.step_length, self.batch_size, replay_buffer_size=self.buffer_size)
        self.dqn.copy_weights_to_target_dqn()
        self.dqn.episode_length = self.episode_length
        self.dqn.steps_copy_target = self.episode_length

        # Share access to the same replay_buffer
        self.dqn.replay_buffer = self.replay_buffer

        self.random_exploration_epsilon = 1
        self.fully_random = True

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
            self.fully_random = True
            return True
        else:
            return False

    # THIS GETS CALLED FIRST HERE NEED TO IMPLEMENT EPSILON GREEDY
    def get_next_action(self, state: np.ndarray):   # TODO REMOVE IS GREEDY FROM RETURN
        # RANDOM EXPLORATION IN BEGINNING
        is_greedy = False # TODO REMOVE IS GREEDY FROM RETURN
        if self.num_steps_taken < self.steps_made_in_exploration:
            self.episode_length = self.random_exploration_episode_length
            if self.episode_counter < 6: # Start at 1
                if self.num_steps_taken % self.episode_length < 50 and not self.got_stuck:
                    action = self.episode_counter + 1 # EPISODE COUNTER STARTS AT 1
                    # print("FOLLOWING")
                    if self.got_stuck:
                        action = np.random.randint(8)
                        while action == self.episode_counter + 1:
                            action = np.random.randint(8)
                else:
                    # 8 actions
                    action = np.random.randint(8)
                    # 4 actions
                    # action = np.random.randint(4)
            else:
                # 8 actions
                action = np.random.randint(8)
                # 4 actions
                # action = np.random.randint(4)
        else:
            self.episode_length = self.actual_episode_length
            action, is_greedy = self.dqn.epsilon_greedy_policy(self.dqn.return_greedy_action(state)) # TODO REMOVE IS GREEDY FROM RETURN

        # Store the action; this will be used later, when storing the transition STORE AS INT
        self.action = action

        action = np.array(self.actions[action])
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition STORE AS LIST EFFICIENCY
        self.state = state # NP ARRAY
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
            reward = 0.3
        else:
            reward = 0

        print("reward", reward)

        transition = (self.state, self.action, reward, next_state)
        self.replay_buffer.add(transition)
        # Add new weight of 1 for the newest transition, we will make sure this gets picked manually by adding to batch
        # Cannot make this 0 for some reason will give error ValueError: probabilities contain NaN
        self.replay_buffer.transition_td_errors.append(0.0001)

        # Train
        if self.num_steps_taken > self.training_threshhold:
            self.dqn.train_q_network_batch(self.replay_buffer.generate_batch(self.batch_size), self.num_steps_taken, self.got_stuck)

        # CROSS ENTROPY METHOD
    def get_greedy_action(self, state: np.ndarray):
        return self.actions[self.dqn.return_greedy_action(state)]


# The DQN class determines how to train the above neural network.
class DQN:
    gamma = 1
    # The class initialisation function.
    def __init__(self, step_length, batch_size, replay_buffer_size, angles_between_actions=2):
        # Create a Q-network, which predicts the q-value for a particular state.
        # 8 actions
        self.q_network = Network(input_dimension=4, output_dimension=1)
        self.target_q_network = Network(input_dimension=4, output_dimension=1)
        # 4 dimensions
        # self.q_network = Network(input_dimension=2, output_dimension=4)
        # self.target_q_network = Network(input_dimension=2, output_dimension=4)

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Step size for each step
        self.step_length = step_length  # TODO here decide whether to normalise and if what size of normalisation

        # Batch size used in the replay_buffer
        self.batch_size = batch_size # TODO update if change the batch size in replay buffer
        self.replay_buffer_size = replay_buffer_size # TODO FEED IN REPLAY BUFFER DIRECTLY

        # Access to the same replay buffer
        self.replay_buffer = None

        # Epsilon
        self.epsilon = 1
        self.steps_increase_epsilon = 5

        # Episode length
        self.episode_length = None
        self.steps_copy_target = self.episode_length
        self.episode_counter = 0

        # is greedy
        self.is_greedy = False
        self.free_steps_taken = 0
        self.greedy_stuck_steps_taken = 0
        self.epsilon_maxed = False


        # # Epsilon linear in episode length
        self.epsilon_increase = 0.005
        self.start_epsilon = 1
        self.standard_epsilon_delta = False

        self.test_actions = None
        self.create_sample_test_steps()

        self.actions = torch.tensor([[-0.02, 0],
                                 [-0.01414, -0.01414],
                                 [0, -0.02],
                                 [0.01414, -0.01414],
                                 [0.02, 0],
                                 [0.01414, 0.01414],
                                 [0, 0.02],
                                 [-0.01414, 0.01414],
                                 ])

        self.state_actions = torch.tensor(np.empty((8,4)))
        self.state_actions[:, [2, 3]] = self.actions

    def create_sample_test_steps(self):
        radians = np.deg2rad(np.array(np.arange(-180, 180, 2)))
        x_steps = np.cos(radians) * self.step_length
        y_steps = np.sin(radians) * self.step_length

        self.test_actions = np.empty((len(radians), 2))
        self.test_actions[:, 0] = x_steps
        self.test_actions[:, 1] = y_steps

    def copy_weights_to_target_dqn(self, other_dqn = False):
        if other_dqn:
            self.q_network.load_state_dict(other_dqn.q_network.state_dict())
        else:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def epsilon_greedy_policy(self, greedy_action, epsilon=False):  # TODO REMOVE IS GREEDY FROM RETURN
        # RANDOM
        print("eps", self.epsilon)
        if np.random.randint(0, 100) in range(int(epsilon * 100)):
            action = self.test_actions[np.random.randint(180)]
            self.is_greedy = False
            return np.array(action), False  # TODO REMOVE IS GREEDY FROM RETURN
        # GREEDY
        else:
            self.is_greedy = True
            return greedy_action, True  # TODO REMOVE IS GREEDY FROM RETURN


    def train_q_network_batch(self, transitions: tuple, step_number, got_stuck):
        if step_number % self.steps_copy_target == 0:
            self.copy_weights_to_target_dqn()

        # increase epsilon later as we go through episodes and hopefully know more about the initial areas
        if step_number % self.episode_length == 0:
            self.episode_counter += 1
            self.epsilon_increase = 0.003
            self.steps_increase_epsilon += 5

        self.optimiser.zero_grad()
        tensor_state_actions, tensor_rewards, tensor_next_states, buffer_indices = transitions
        # Network predictions is a *x1 tensor of the current state action values in the batch
        tensor_current_state_action_value = self.q_network.forward(tensor_state_actions)



        # Given the next state, we want to find the greedy action in the next state and use it to compute the next state's value
        # This uses the target network
        tensor_next_states_values = self.return_next_state_values_tensor(tensor_next_states)
        tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_next_states_values
        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_predicted_q_value_current_state)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        td_error = np.abs((tensor_bellman_current_state_value - tensor_predicted_q_value_current_state).detach().numpy()).ravel()

        print("td", step_number, np.mean(td_error))
        print(self.episode_counter)

        # Update the TD errors of the transitions we used
        for index, error in zip(buffer_indices, td_error):
            self.replay_buffer.transition_td_errors[index] = error

        step_in_episode = step_number % self.episode_length
        episode_number = step_number // self.episode_length
        within_episode_scale = step_in_episode / self.episode_length

        # PURELIENAR
        if step_in_episode > self.steps_increase_epsilon:
            self.epsilon += self.epsilon_increase

        if self.episode_length - step_in_episode < 20:
            self.epsilon += 0.1
        if self.episode_length - step_in_episode < 6:
            self.epsilon = 0.2

        if episode_number % 10 == 1:
            self.standard_epsilon_delta = True
            self.epsilon = self.start_epsilon
        elif episode_number % 5 == 0:
            self.standard_epsilon_delta = False

        # Standard Epsilon Delta
        if self.standard_epsilon_delta:
            self.epsilon -= 0.003

        if not self.standard_epsilon_delta:
            # Set epsilon at start of episode
            if step_in_episode == 1:
                self.epsilon = 0.2
                self.epsilon_maxed = False
            #
            # PURELIENAR
            if step_in_episode > self.steps_increase_epsilon:
                self.epsilon += self.epsilon_increase

            if self.episode_length - step_in_episode < 40:
                self.epsilon += 0.1
            if self.episode_length - step_in_episode < 10:
                self.epsilon = 0.2

        self.epsilon = min(1, self.epsilon)
        self.epsilon = max(0, self.epsilon)

        # # LEARNING RATE UPDATE TODO with starting rate at 0.003
        # if self.episode_counter == 10:
        #     self.optimiser.param_groups[0]["lr"] = 0.002
        # elif self.episode_counter == 12:
        #     self.optimiser.param_groups[0]["lr"] = 0.001
        # elif self.episode_counter == 17:
        #     self.optimiser.param_groups[0]["lr"] = 0.0005

        return loss.item()

    def return_greedy_action(self, current_state):
        self.state_actions[:, [0, 1]] = torch.tensor(np.tile(current_state, reps=(8,1)))
        # input_tensor = torch.tensor(current_state).float().unsqueeze(0)

        network_prediction = self.q_network.forward(self.state_actions)
        return int(network_prediction.argmax())


    def return_next_state_values_tensor(self, tensor_next_states):
        # Double Q

        tensor_network_predictions = self.q_network.forward(tensor_next_states)
        predictions_np_array = tensor_network_predictions.detach().numpy()
        greedy_actions = np.argmax(predictions_np_array, axis=1) # TODO
        # Reshape from 1D 1x* to 2D *x 1 array so can transform and output a tensor
        tensor_greedy_actions = torch.tensor(greedy_actions.reshape(-1, 1))
        tensor_next_state_values = torch.gather(tensor_network_predictions, 1, tensor_greedy_actions)
        return tensor_next_state_values

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
        min_probability_constant = 0.05

        # Normalise weights
        weights = (np.array(self.transition_td_errors) + min_probability_constant) / (
                    sum(self.transition_td_errors) + min_probability_constant * self.length)

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


if __name__ == '__main__':
    dqn = DQN(10, 10, replay_buffer_size=10)
    print(dqn.optimiser.param_groups[0]["lr"])
