import numpy as np
from ForMaze import get_states_from_Maze,first_visit_mc_prediction,on_policy_first_visit_mc_control, mc_control_es, td_zero, sarsa
from Maze_generating_interface import App

app = App()
app.mainloop()
Maze = app.A

#states, exit_state , init_states = get_states_from_Maze(Maze)

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.maze_size = maze.shape
        self.exit_state = tuple(np.argwhere(maze == 2)[0])
        self.current_state = None

    def reset(self):
        init_states = np.argwhere(self.maze == 0)
        self.current_state = tuple(init_states[np.random.randint(len(init_states))])
        return self.current_state

    def step(self, action):
        next_state = np.array(self.current_state)
        if action == 0:  # UP
            next_state[0] -= 1
        elif action == 1:  # RIGHT
            next_state[1] += 1
        elif action == 2:  # DOWN
            next_state[0] += 1
        elif action == 3:  # LEFT
            next_state[1] -= 1

        # Check maze boundaries
        if next_state[0] < 0 or next_state[0] >= self.maze_size[0] or \
           next_state[1] < 0 or next_state[1] >= self.maze_size[1]:
            next_state = np.array(self.current_state)

        # If the next cell is a wall stay in the current state
        elif self.maze[tuple(next_state)] == 1:
            next_state = np.array(self.current_state)
        else:
            self.current_state = tuple(next_state)

        done = np.array_equal(self.current_state, self.exit_state)
        reward = 1 if done else -0.01 
        return self.current_state, reward, done, {}

env = MazeEnv(Maze)

Q_mc, policy_mc = mc_control_es(env, num_episodes=1000, epsilon=0.7)
print("\nThis is value function of monte carlo exploring start:")
for state, reward in Q_mc.items():
    print(f"The Curent state: {state[0]}, Action : {state[1]}, Reward: {reward}")

print("\nMonte Carlo Control (Exploring Starts) Optimal Policy:")
for state, action in policy_mc.items():
    print(f"When You are in State: {state}, Best Action to take: {action}")


V_td = td_zero(env, num_episodes=500)
print("\nTD[0] Prediction - State Value Function:")
for state, value in V_td.items():
    print(f"When You are in State: {state}, Value to take: {value}")



Q_sarsa, policy_sarsa = sarsa(env, num_episodes=500, epsilon=0.5)
print("\nThis is reward fucntion of sarsa:")
for state, reward in Q_sarsa.items():
    print(f"The Curent state: {state[0]}, Action : {state[1]}, Reward: {reward}")
    
print("\nSARSA - Policy:")
for state, action in policy_sarsa.items():
    print(f"When your are in state: {state}, Best Action to take: {action}")

