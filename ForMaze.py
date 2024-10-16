import numpy as np
import random
from collections import defaultdict
# 0 : UP / 1 : right / 2: down / 3: Left
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def get_actions(state, Maze):
    local_actions = []
    for a in actions:
        m = [0, 0]
        if a == "UP":
            m[0] = -1
        elif a == "DOWN":
            m[0] = +1
        elif a == "LEFT":
            m[1] = -1
        else:
            m[1] = +1
        if Maze[state[0] + m[0], state[1] + m[1]] == 1.0:
            local_actions.append(a)
    return local_actions

def next_state_old(state, action, exit_state, Maze):
    new_state = np.copy(state)
    if action == "UP":
        new_state[0] -= 1
    elif action == "DOWN":
        new_state[0] += 1
    elif action == "LEFT":
        new_state[1] -= 1
    else:
        new_state[1] += 1

    reward = -0.01
    if all(new_state == exit_state):
        reward = 1
    return new_state, reward

def next_state(state, action, exit_state, Maze):
    new_state = np.copy(state)

    # Update the state based on the action taken
    if action == "UP":
        new_state[0] -= 1
    elif action == "DOWN":
        new_state[0] += 1
    elif action == "LEFT":
        new_state[1] -= 1
    elif action == "RIGHT":
        new_state[1] += 1

    # Define boundaries of the maze
    if new_state[0] < 0 or new_state[0] >= Maze.shape[0] or new_state[1] < 0 or new_state[1] >= Maze.shape[1]:
        # If out of bounds, revert to the original state and apply a penalty
        new_state = state
        reward = -1  # Penalty for hitting the wall or going out of bounds
    elif Maze[new_state[0], new_state[1]] == 0:
        # If the new state is a wall (assuming walls are marked with 0)
        new_state = state
        reward = -1  # Penalty for hitting a wall
    else:
        # Default negative reward for each action taken
        reward = -0.01
        
        # Check if the new state is the exit state
        if np.array_equal(new_state, exit_state):
            reward = 1  # Reward for reaching the exit

    return new_state, reward

def generate_episode(init_state, Maze, exit_state, itermax=3000):
    episode = []
    current_state = np.copy(init_state)
    i = 0
    while i < itermax and not all(current_state == exit_state):
        current_actions = get_actions(current_state, Maze)
        action = current_actions[np.random.randint(len(current_actions))]
        new_state, reward = next_state(current_state, action, exit_state, Maze)
        episode.append((current_state, action, reward, new_state))
        current_state = new_state
        i += 1
    return episode

def get_states_from_Maze(Maze):
    states = list(np.argwhere(Maze == 1))
    exit_state = np.argwhere(Maze == 2)[0]
    init_states = list(np.copy(states))
    if np.any((init_states == exit_state).sum(axis=1) == 2):
        init_states.pop(np.argwhere((init_states == exit_state).sum(axis=1) == 2)[0][0])
    return states, exit_state, init_states

def get_states_from_Maze_NEW(Maze):
    # Get all wall states (cells with value 1)
    states = list(np.argwhere(Maze == 1))
    
    # Get the exit state (cell with value 2)
    exit_state = np.argwhere(Maze == 2)[0]
    
    # Get the start state (cell with value -1)
    start_state = np.argwhere(Maze == -1)[0]
    
    # Initialize init_states excluding both the start and exit states
    init_states = list(np.copy(states))
    
    # Remove the exit state from init_states, if it exists in walls (for consistency)
    if np.any((init_states == exit_state).sum(axis=1) == 2):
        init_states.pop(np.argwhere((init_states == exit_state).sum(axis=1) == 2)[0][0])

    # Remove the start state from init_states, if it exists in walls (for consistency)
    if np.any((init_states == start_state).sum(axis=1) == 2):
        init_states.pop(np.argwhere((init_states == start_state).sum(axis=1) == 2)[0][0])

    return states, exit_state,start_state, init_states



""" Monte Carlo Algorithms """
def mc_control_es(env, num_episodes, epsilon, discount_factor=1.0):
    Q = {}
    returns_sum = {}
    returns_count = {}

    for x in range(env.maze_size[0]):
        for y in range(env.maze_size[1]):
            for a in range(4):
                Q[((x, y), a)] = 0.0
                returns_sum[((x, y), a)] = 0.0
                returns_count[((x, y), a)] = 0.0

    def epsilon_greedy_policy(Q, state, epsilon=0.3):
        if random.random() < epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            return max([a for a in range(4)], key=lambda action: Q.get((state, action), 0.0))

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        action = random.choice([0, 1, 2, 3])
        episode = []

        for t in range(100):
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            action = epsilon_greedy_policy(Q, state, epsilon)

        state_action_pairs = set([(x[0], x[1]) for x in episode])
        for state, action in state_action_pairs:
            first_occurence_idx = next(i for i, x in enumerate(episode) if (x[0], x[1]) == (state, action))
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]
    policy = {}
    for state in Q:
        state_only = state[0]

    # Check if the state is a wall
        if env.maze[state_only] == 1:
           policy[state_only] = "00"  # Wall state, set action to "00"
    
    # Check if the state is the exit
        elif np.array_equal(state_only, env.exit_state):
           policy[state_only] = "11"  # Exit state, set action to "11"
    
        else:
        # For regular open states, continue with normal logic to find the best action
           best_action = max([a for a in range(4)], key=lambda a: Q.get((state_only, a), 0.0))
           policy[state_only] = best_action

    return Q, policy

def first_visit_mc_prediction(env, num_episodes, discount_factor=1.0):
    returns_sum = {}
    returns_count = {}
    V = {}
    policy = {}

    for _ in range(num_episodes):
        episode = generate_episode(env.maze, env.exit_state)
        seen_states = set()
        
        for i, (state, action, reward, _) in enumerate(episode):
            if state not in seen_states:
                # Calculate the return G from this state onward
                G = sum([discount_factor**t * r for t, (_, _, r, _) in enumerate(episode[i:])])
                returns_sum[state] += G
                returns_count[state] += 1
                # Update state-value function
                V[state] = returns_sum[state] / returns_count[state]
                seen_states.add(state)
                
    return V

def on_policy_first_visit_mc_control(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    Q = {} 
    returns_sum = {}
    returns_count = {}
    
    for _ in range(num_episodes):
        # Randomly choose an initial state
        state = env.init_state
        episode = generate_episode(state,env.maze, env.exit_state)
        seen_state_actions = set()
        
        for i, (state, action, reward, _) in enumerate(episode):
            if (state, action) not in seen_state_actions:
                # Calculate the return G from this state-action pair onward
                G = sum([discount_factor**t * r for t, (_, _, r, _) in enumerate(episode[i:])])
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                # Update action-value function
                action_idx = ['UP', 'DOWN', 'LEFT', 'RIGHT'].index(action)
                Q[state][action_idx] = returns_sum[(state, action)] / returns_count[(state, action)]
                seen_state_actions.add((state, action))
        
        # Update the policy with an epsilon-greedy approach
        for state in Q:
            best_action_idx = np.argmax(Q[state])
            for a in range(4):  # Four possible actions
                if a == best_action_idx:
                    Q[state][a] = 1 - epsilon + (epsilon / 4)
                else:
                    Q[state][a] = epsilon / 4
    
    return Q

""" Temporal diffrence """
def td_zero(env, num_episodes, discount_factor=1.0, alpha=0.1):
    V = {}
    for x in range(env.maze_size[0]):
        for y in range(env.maze_size[1]):
            V[(x, y)] = 0.0

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        for t in range(100):
            action = random.choice([0, 1, 2, 3])
            next_state, reward, done, _ = env.step(action)
            V[state] = V[state] + alpha * (reward + discount_factor * V[next_state] - V[state])
            if done:
                break
            state = next_state

    return V

def sarsa(env, num_episodes, discount_factor=1.0, epsilon=0.3, alpha=0.1):
    Q = {}
    for x in range(env.maze_size[0]):
        for y in range(env.maze_size[1]):
            for a in range(4):
                Q[((x, y), a)] = 0.0

    def epsilon_greedy_policy(Q, state, epsilon=0.3):
        if random.random() < epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            return max([a for a in range(4)], key=lambda action: Q.get((state, action), 0.0))

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)

        for t in range(100):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            Q[(state, action)] = Q[(state, action)] + alpha * (reward + discount_factor * Q[(next_state, next_action)] - Q[(state, action)])
            if done:
                break
            state = next_state
            action = next_action

    policy = {}
    for state in Q:
        state_only = state[0]

    # Check if the state is a wall
        if env.maze[state_only] == 1:  
           policy[state_only] = "00"  
    
    # Check if the state is the exit
        elif np.array_equal(state_only, env.exit_state):  
           policy[state_only] = "11"  
    
        else:
           best_action = max([a for a in range(4)], key=lambda a: Q.get((state_only, a), 0.0))
           policy[state_only] = best_action


    return Q, policy