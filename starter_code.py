# Import some modules from other libraries
import numpy as np
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import collections

from environment import Environment
from q_visualisation import QVisualisation
from path_visualisation import PathVisualisation


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
        self.epsilon = 1.


    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, action: int = False):
        # Choose an action.
        if action is not False:
            discrete_action = action
        else:
            discrete_action = np.random.randint(0, 4)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this action.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def step_alt_reward(self, action: int = False):
        # Choose an action.
        if action is not False:
            discrete_action = action
        else:
            discrete_action = np.random.randint(0, 4)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, _ = self.environment.step(self.state, continuous_action)
        reward = -99
        # IMPLEMENT NEW REWARD FUNCTION, MANHATTAN DISTANCE
        # reward = 1.7 - (abs(next_state[0] - self.environment.goal_state[0]) + abs(next_state[1] - self.environment.goal_state[1]))
        # MAX DIST
        # reward = 1 - (max(abs(next_state[0] - self.environment.goal_state[0]), abs(
        #     next_state[1] - self.environment.goal_state[1])))
        # UP AND RIGHT
        # REWARD ACTIONS
        # If standing still
        if np.linalg.norm(next_state - self.state) < 0.0001 and self.return_final_distance(next_state, "e") > 0.15:
            # scale negative reward for standing still by the distance from goal state
            reward = -20
        # Right
        elif action in {0}:
            reward = 4
        # Left or down
        elif action in {1, 3}:
            reward = -5
        # Up
        elif action in {2}:
            reward = 20

        # REWARD DISTANCE
        # If standing still
        x_movement = self.return_x_distance(self.state) - self.return_x_distance(next_state)
        y_movement = self.return_y_distance(self.state) - self.return_y_distance(next_state)

        if np.linalg.norm(next_state - self.state) < 0.0001 and self.return_final_distance(next_state, "e") > 0.15:
            # scale negative reward for standing still by the distance from goal state
            reward = -2
        # if distance increases, then penalise
        elif self.return_final_distance(next_state) > self.return_final_distance(self.state):
            reward = -1 * self.return_final_distance(next_state)
        # if the state is closer on x than on y
        elif self.return_x_distance(self.state) <= self.return_y_distance(self.state):
            # if move closer, get reward 1, if not move, 0, if away get -1
            reward = 10 * x_movement
        elif self.return_x_distance(self.state) > self.return_y_distance(self.state):
            reward = 10 * y_movement

        # Reward not standing still VIABLE
        if np.linalg.norm(next_state - self.state) < 0.0001 and self.return_final_distance(next_state, "e") > 0.15:
            # scale negative reward for standing still by the distance from goal state
            reward = -0.2
        # if distance increases, then penalise
        else:
            reward = 1 - np.linalg.norm(next_state - self.environment.goal_state)

        # Reward not standing still alt2
        # if np.linalg.norm(next_state - self.state) < 0.0001 and self.return_final_distance(next_state, "e") > 0.15:
        #     # scale negative reward for standing still by the distance from goal state
        #     reward = -1
        # # if distance increases, then penalise
        # else:
        #     reward = np.linalg.norm(next_state - self.environment.goal_state)


        # reward = -np.linalg.norm(next_state - self.goal_state)

        # reward = -np.linalg.norm(next_state - self.goal_state)


        # ADD A LINE WHERE WE ARE FULLY GREEDY IE WHEN EPSILON = 0

        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def return_final_distance(self, state, metric="m"):
        if metric =="m":
            return abs(state[0] - self.environment.goal_state[0]) + abs(state[1] - self.environment.goal_state[1])
        elif metric == "e":
            return np.linalg.norm(state - self.environment.goal_state)

    def return_x_distance(self, state):
        return abs(state[0] - self.environment.goal_state[0])

    def return_y_distance(self, state):
        return abs(state[1] - self.environment.goal_state[1])

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

    def epsilon_greedy_policy(self, greedy_action):
        if np.random.randint(0, 100) in range(int(self.epsilon * 100)):
            return np.random.randint(0, 4)
        else:
            return greedy_action

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
    gamma = 0.9
    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_q_network = Network(input_dimension=2, output_dimension=4)
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
        tensor_current_states, tensor_actions, tensor_rewards, tensor_next_states = transitions
        # Network predictions is a *x4 tensor of 4 state value predictions per row, one for each action
        network_predictions = self.q_network.forward(tensor_current_states)
        tensor_predicted_q_value_current_state = torch.gather(network_predictions, 1, tensor_actions)

        # Given the next state, we want to find the greedy action in the next state and use it to compute the next state's value
        # This will now use the target_q_network
        tensor_next_states_values = self.return_next_state_values_tensor(tensor_next_states)
        tensor_bellman_current_state_value = tensor_rewards + self.gamma * tensor_next_states_values
        loss = torch.nn.MSELoss()(tensor_bellman_current_state_value, tensor_predicted_q_value_current_state)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

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
        # print(network_prediction)
        # print(int(network_prediction.argmax()))
        return int(network_prediction.argmax())

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
    def __init__(self, max_capacity=1000000):
        self.replay_buffer = collections.deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, transition_tuple):
        self.replay_buffer.append(transition_tuple)

    def clear(self):
        self.replay_buffer.clear()

    # Returns tuple of tensors, each has dimension (batch_size, *), SARS'
    def generate_batch(self, batch_size=50):
        current_states = []
        actions = []
        rewards = []
        next_states = []
        for _ in range(batch_size):
            transition = self.replay_buffer[np.random.randint(len(self))]
            current_states.append(transition[0])  # 1x2
            actions.append([transition[1]])  # 1x1
            rewards.append([transition[2]])  # 1x1
            next_states.append(transition[3])  # 1x2
        return torch.tensor(current_states), torch.tensor(actions), torch.tensor(rewards).float(), torch.tensor(
            next_states)  # MSE needs float values, so cast rewards to floats

# Main entry point
if __name__ == "__main__":
    plot_loss = True
    plot_qvalues = False
    plot_state_path = False
    # Set the random seed for both NumPy and Torch
    CID = 741321
    np.random.seed(CID)
    torch.manual_seed(CID)
    # initialise both environments and agents
    environment_old = Environment(display=False, magnification=1000)
    agent_old = Agent(environment_old)
    environment_new = Environment(display=False, magnification=1000)
    agent_new = Agent(environment_new)
    # Create a ReplayBuffer and batch size
    replay_buffer_old = ReplayBuffer()
    replay_buffer_new = ReplayBuffer()
    rb_batch_size = 50
    optimal_delta = 0.0032
    dqn_old = DQN()
    dqn_new = DQN()
    # Make sure both networks have the same initial weights as the random seed cannot be reinitialised now
    dqn_new.copy_weights_to_target_dqn(dqn_old)
    dqn_old.copy_weights_to_target_dqn()
    dqn_new.copy_weights_to_target_dqn()

    
    # have 2 networks, one is trained with the normal reward function, one with the new one
    # after each step, based on the training, have both networks run greedily for one episode (length 20) and find the final states distance
    # append these distances to a list, so have a list of 500 after all the training
    # create test agent and test environment for each trial

    episode_counter = 0
    total_steps_counter = 0

    distances_old = []
    distances_new = []
    while True:
        if episode_counter == 25:
            break
        episode_counter += 1
        # Reset the environment for the start of the episode.
        agent_old.reset()
        agent_new.reset()
        # Loop over steps within this episode.
        for step_num in range(20):
            # Every 20 steps update target DQN
            if total_steps_counter % 20 == 0:
                dqn_old.copy_weights_to_target_dqn()
                dqn_new.copy_weights_to_target_dqn()
            
            # For both environments, make the transition                
            # Old reward
            greedy_action_old = dqn_old.return_greedy_action(agent_old.state)
            epsilon_greedy_action_old = agent_old.epsilon_greedy_policy(greedy_action_old)
            transition_old = agent_old.step(epsilon_greedy_action_old)
            # This is with the new reward
            greedy_action_new = dqn_new.return_greedy_action(agent_new.state)
            epsilon_greedy_action_new = agent_new.epsilon_greedy_policy(greedy_action_new)
            transition_new = agent_new.step_alt_reward(epsilon_greedy_action_new)
            # print(f"greedy {greedy_action}, epsilongree {epsilon_greedy_action}, epsi {agent.epsilon}")
            # print(dqn.q_network.forward(torch.tensor(current_state).unsqueeze(0))) # print qvalue predictions
            
            # print(transition)
            # Update epsilon after each step
            if agent_old.epsilon > 0:
                # print(agent.epsilon)
                # Lower bound of epsilon is 0, technically not necessary with the random implementation
                agent_old.epsilon = max(agent_old.epsilon - optimal_delta, 0)
                agent_new.epsilon = agent_old.epsilon

            # Add transitions to replay buffers
            replay_buffer_old.add(transition_old)
            replay_buffer_new.add(transition_new)

            # Training using target network
            if len(replay_buffer_old) < rb_batch_size:
                train_network = False
            else:
                train_network = True

            if train_network:
                loss_old = dqn_old.train_q_network_batch(replay_buffer_old.generate_batch(rb_batch_size))
                loss_new = dqn_new.train_q_network_batch(replay_buffer_new.generate_batch(rb_batch_size))

            # Here run full episode with greedy policy
            # initialise a new environment and agent for both networks
            environment_test_old = Environment(display=False, magnification=1000)
            agent_test_old = Agent(environment_test_old)
            environment_test_new = Environment(display=False, magnification=1000)
            agent_test_new = Agent(environment_test_new)
            for step in range(20):
                greedy_action_test_old = dqn_old.return_greedy_action(agent_test_old.state)
                transition_old = agent_test_old.step(greedy_action_test_old)
                greedy_action_test_new = dqn_new.return_greedy_action(agent_test_new.state)
                transition_new = agent_test_new.step_alt_reward(greedy_action_test_new)
                if episode_counter > 23: # DEBUG TODO
                    print(transition_new)
            print("test episode done")
            # print(agent_test_old.state)
            # print(agent_test_new.state)

            final_distance_old = agent_test_old.return_final_distance(agent_test_old.state, "e")
            final_distance_new = agent_test_new.return_final_distance(agent_test_new.state, "e")
            # print(final_distance_old)
            # print(final_distance_new)
            distances_old.append(final_distance_old)
            distances_new.append(final_distance_new)

            total_steps_counter += 1

    difference = np.array(distances_old) - np.array(distances_new)
    differences = [round(diff, 3) for diff in difference]
    print(differences)
    print(distances_old)
    print(distances_new)
    #
    # # Plotting the loss functions as function of steps and time
    if plot_loss:
        # Delta axis
        plt.plot(range(500), distances_old, color="green", label="old")
        # ax2 = ax1.twiny()
        # ax1.set_xlabel("Delta value")
        plt.plot(range(500), distances_new, color="red")
        plt.legend()

        plt.ylabel("Final distance")

        # Add vertical lines
        # for step_num in range(500, len(losses) + 500, 20):
        #     ax1.axvline(step_num, ls="--")
        plt.show()

        # start both x axis on 0?
    #
    # # steps of 0.05 as each state is 0.1 distance away, know from the obstacle
    # if plot_qvalues:
    #     # Because CV plots from top to bottom, origin is top left, we start with the upper row of states
    #     states_x_coords = np.arange(0.05, 1, 0.1)
    #     states_y_coords = np.arange(0.95, 0, -0.1)
    #
    #     colour_factors = []
    #     for y_coord in states_y_coords:
    #         for x_coord in states_x_coords:
    #             input_tensor = torch.tensor([[x_coord, y_coord]])
    #             colour_factors.append(dqn.return_optimal_action_order(input_tensor))
    #
    #     qv = QVisualisation(1000)
    #     qv.draw(colour_factors)
    #     time.sleep(15)
    #
    if plot_state_path:
        state_path_old = []
        agent_old.reset()
        state_path_new = []
        agent_new.reset()
        # Loop over steps within this episode.
        for step_num in range(20):
            # Take the greedy action step to plot the state path
            current_state = agent_old.state
            state_path_old.append(current_state)
            greedy_action = dqn_old.return_greedy_action(current_state)
            transition_old = agent_old.step(greedy_action)
            current_state = agent_new.state
            state_path_new.append(current_state)
            greedy_action = dqn_new.return_greedy_action(current_state)
            transition_new = agent_new.step(greedy_action)
            print(transition_old)
            print(transition_new)

        pv_old = PathVisualisation(1000)
        pv_old.draw(state_path_old, True, True)
        time.sleep(15)
        pv_new = PathVisualisation(1000)
        pv_new.draw(state_path_new, True, True)
        time.sleep(15)