
import numpy as np
from collections import defaultdict

# First-Visit MC Prediction Algorithm
def first_visit_mc_prediction(policy, env, num_episodes, gamma=1.0):
    V = defaultdict(float)  # State-value function
    returns = defaultdict(list)  # Store returns for each state

    for _ in range(num_episodes):
        episode = generate_episode(env, policy)  # Function that generates episodes
        visited_states = set()
        for i, (state, _, reward) in enumerate(episode):
            if state not in visited_states:
                visited_states.add(state)
                G = sum([reward * (gamma**t) for t, (_, _, reward) in enumerate(episode[i:])])
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    return V

# Monte Carlo Exploring Starts Algorithm
def mc_exploring_starts(env, num_episodes, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    
    for _ in range(num_episodes):
        state = env.reset()
        action = np.random.choice(env.action_space.n)  # Exploring start
        episode = generate_episode(env, state, action)  # Use given state and action
        visited_state_action_pairs = set()
        for i, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited_state_action_pairs:
                visited_state_action_pairs.add((state, action))
                G = sum([r * (gamma**t) for t, (_, _, r) in enumerate(episode[i:])])
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
    policy = {state: np.argmax(Q[state]) for state in Q}
    return Q, policy

# On-Policy First-Visit MC Control Algorithm
def on_policy_first_visit_mc_control(env, num_episodes, epsilon=0.1, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    
    def epsilon_greedy_policy(state, Q, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(env.action_space.n)
        else:
            return np.argmax(Q[state])

    for _ in range(num_episodes):
        episode = generate_episode(env, Q, epsilon_greedy_policy)  # Generate episode using ε-soft policy
        visited_state_action_pairs = set()
        for i, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited_state_action_pairs:
                visited_state_action_pairs.add((state, action))
                G = sum([reward * (gamma**t) for t, (_, _, reward) in enumerate(episode[i:])])
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
    policy = {state: np.argmax(Q[state]) for state in Q}
    return Q, policy

# Off-Policy MC Prediction Algorithm
def off_policy_mc_prediction(env, target_policy, behavior_policy, num_episodes, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(num_episodes):
        episode = generate_episode(env, behavior_policy)
        G = 0
        W = 1
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += W / C[state][action] * (G - Q[state][action])
            if action != target_policy[state]:
                break
            W *= 1 / behavior_policy(state)[action]
    return Q

# Off-Policy MC Control Algorithm
def off_policy_mc_control(env, behavior_policy, num_episodes, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(num_episodes):
        episode = generate_episode(env, behavior_policy)
        G = 0
        W = 1
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += W / C[state][action] * (G - Q[state][action])
            W *= 1 / behavior_policy(state)[action]
    policy = {state: np.argmax(Q[state]) for state in Q}
    return Q, policy

# TD(0) Prediction Algorithm
def td_zero_prediction(policy, env, num_episodes, alpha=0.1, gamma=1.0):
    V = defaultdict(float)
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    return V

# SARSA Algorithm
def sarsa(env, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def epsilon_greedy_policy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(env.action_space.n)
        else:
            return np.argmax(Q[state])

    for _ in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(state)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(next_state)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
    return Q

# Q-Learning Algorithm
def q_learning(env, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def epsilon_greedy_policy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(env.action_space.n)
        else:
            return np.argmax(Q[state])

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_policy(state)
            next_state, reward, done, _ = env.step(action)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
    return Q
