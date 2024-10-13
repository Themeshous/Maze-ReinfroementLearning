import numpy as np
from ForMaze import get_states_from_Maze, mc_control_es, td_zero, sarsa
from Maze_generating_interface import App

app = App()
app.mainloop()
Maze = app.A

states, exit_state, init_states = get_states_from_Maze(Maze)

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.maze_size = maze.shape
        self.exit_state = np.argwhere(maze == 2)[0]
        self.current_state = None

    def reset(self):
        init_states = np.argwhere(self.maze == 1)
        self.current_state = tuple(init_states[np.random.randint(len(init_states))])
        return self.current_state

    def step(self, action):
        next_state = np.array(self.current_state)
        if action == 0:
            next_state[0] -= 1
        elif action == 1:
            next_state[1] += 1
        elif action == 2:
            next_state[0] += 1
        elif action == 3:
            next_state[1] -= 1

        if next_state[0] < 0 or next_state[0] >= self.maze_size[0] or next_state[1] < 0 or next_state[1] >= self.maze_size[1]:
            next_state = np.array(self.current_state)

        if self.maze[tuple(next_state)] == 1:
            self.current_state = tuple(next_state)
        else:
            next_state = np.array(self.current_state)

        done = np.array_equal(self.current_state, self.exit_state)
        reward = -0.01
        if done:
            reward = 1

        return self.current_state, reward, done, {}

env = MazeEnv(Maze)

Q_mc, policy_mc = mc_control_es(env, num_episodes=100, epsilon=0.3)
print("Monte Carlo Control (Exploring Starts) - Policy:")
for state, action in policy_mc.items():
    print(f"State: {state}, Best Action: {action}")

V_td = td_zero(env, num_episodes=100)
print("\nTD(0) Prediction - State Value Function:")
for state, value in V_td.items():
    print(f"State: {state}, Value: {value}")

Q_sarsa, policy_sarsa = sarsa(env, num_episodes=100, epsilon=0.3)
print("\nSARSA - Policy:")
for state, action in policy_sarsa.items():
    print(f"State: {state}, Best Action: {action}")
