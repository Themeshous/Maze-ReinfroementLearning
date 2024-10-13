import numpy as np
import random

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

def next_state(state, action, exit_state, Maze):
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

def generate_episode(init_state, Maze, exit_state, itermax=1000):
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

def mc_control_es(env, num_episodes, discount_factor=1.0, epsilon=0.3):
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
        best_action = max([a for a in range(4)], key=lambda a: Q.get((state_only, a), 0.0))
        policy[state_only] = best_action

    return Q, policy

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
        best_action = max([a for a in range(4)], key=lambda a: Q.get((state_only, a), 0.0))
        policy[state_only] = best_action

    return Q, policy
