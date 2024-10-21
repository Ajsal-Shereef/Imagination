import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv

# Define the possible directions the agent can face (4 possible directions: 0-3)
NUM_DIRECTIONS = 4

def get_one_hot_encoding(idx, size):
    """Helper function to create one-hot encoding."""
    one_hot = np.zeros(size)
    one_hot[idx] = 1
    return one_hot

def preprocess_observation(obs):
    """
    Preprocesses the MiniGrid observation using one-hot encoding, including the agent's direction.
    
    Args:
        obs: Observation from the environment, a dictionary with keys `image`, `direction`, and `mission`.
    Returns:
        processed_obs: Preprocessed observation with one-hot encoded features for objects, colors, states, and agent direction, flattened into a 1D vector.
    """
    # Extract the grid (partially observable view) and the agent's direction
    grid_obs = obs['image']  # Shape: (grid_height, grid_width, 3)
    agent_direction = obs['direction']  # Scalar value for direction (0 to 3)

    # Get dimensions for the grid (height, width) and the number of visible cells
    height, width, _ = grid_obs.shape
    num_cells = height * width
    
    # Precompute the sizes for one-hot encoding based on the constants
    num_objects = len(OBJECT_TO_IDX)  # Total number of objects
    num_colors = len(COLOR_TO_IDX)    # Total number of colors
    num_states = len(STATE_TO_IDX)    # Total number of states (open, closed, etc.)
    
    # Create an empty list to store the encoded features
    processed_obs = []

    # Iterate over each cell in the grid (partially observable view)
    for i in range(height):
        for j in range(width):
            # Each cell has an object, color, and state
            obj_type = grid_obs[i, j, 0]
            obj_color = grid_obs[i, j, 1]
            obj_state = grid_obs[i, j, 2]
            
            # One-hot encode the object type, color, and state
            obj_one_hot = get_one_hot_encoding(obj_type, num_objects)
            color_one_hot = get_one_hot_encoding(obj_color, num_colors)
            state_one_hot = get_one_hot_encoding(obj_state, num_states)
            
            # Concatenate the one-hot encoded features for this cell
            encoded_cell = np.concatenate([obj_one_hot, color_one_hot, state_one_hot])
            
            # Append to the list of processed observations
            processed_obs.append(encoded_cell)
    
    # Convert the list to a numpy array and reshape it to match the flattened view of the grid
    processed_obs = np.array(processed_obs).reshape((num_cells, -1))
    
    # One-hot encode the agent's direction and concatenate it to the processed observation
    direction_one_hot = get_one_hot_encoding(agent_direction, NUM_DIRECTIONS)
    
    # Concatenate direction encoding with the processed grid observation
    processed_obs = np.concatenate([processed_obs.flatten(), direction_one_hot])

    return processed_obs

# def preprocess_observation(obs):
#     """
#     Preprocesses the MiniGrid observation using one-hot encoding, including the agent's direction.
    
#     Args:
#         obs: Observation from the environment, a dictionary with keys `image`, `direction`, and `mission`.
#     Returns:
#         processed_obs: Preprocessed observation with one-hot encoded features for objects, colors, states, and agent direction, flattened into a 1D vector.
#     """
#     return np.concatenate([obs['image'].flatten(), get_one_hot_encoding(obs["direction"], NUM_DIRECTIONS)])

# Example usage:
# Assuming `obs` is the observation dictionary received from the environment
# Example grid size could be 7x7 for a partially observable environment
# processed_obs = preprocess_observation(obs, grid_size=7)


class DoorKeyPickup(MiniGridEnv):
    def __init__(
        self,
        max_steps,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Open the blue gate with the blue keys and reach the target"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))
        
        self.grid.set(5, 2, Door(COLOR_NAMES[3], is_locked=True))
        self.grid.set(1, 3, Key(COLOR_NAMES[3]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Open the blue gate with the blue key and reach the target"
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        return obs, reward, terminated, truncated, info
    
class SimplePickup(MiniGridEnv):
    def __init__(
        self,
        max_steps,
        size=10,
        agent_start_pos=None,
        agent_start_dir=None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(504,), dtype=np.float32)

    @staticmethod
    def _gen_mission():
        return "Pick the green ball"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Function to generate a set of two random numbers
        # def generate_random_set():
        #     return (random.randint(1, width-2), random.randint(1, height-2))

        # # Generate the first set
        # set1 = generate_random_set()

        # # Generate the second set and ensure it's not the same as the first set
        # set2 = generate_random_set()
        # while set2 == set1:
        #     set2 = generate_random_set()

        # # Place a ball square in the bottom-right corner
        # self.put_obj(Ball(color='green'), set1[0], set1[1])
        
        # # Place a ball square in the bottom-right corner
        # self.put_obj(Ball(color='red'), set2[0], set2[1])
        
        # Place a ball square in the bottom-right corner
        self.put_obj(Ball(color='green'), 1, width-2)
        
        # Place a ball square in the bottom-right corner
        self.put_obj(Ball(color='red'), width-2, height-2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Pick the green ball"
        
    def reset(self):
        obs, info = super().reset()
        return preprocess_observation(obs), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = preprocess_observation(obs)

        # Check if the agent has picked up the goal object (e.g., the ball)
        if isinstance(self.carrying, Ball) and self.carrying.color == 'green':
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info