import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.minigrid_env import MiniGridEnv

def preprocess_observation(observation):
    """
    Preprocess the MiniGrid observation into a feature-based representation.
    
    Args:
        observation (dict): The observation from the MiniGrid environment.
        num_objects (int): Number of distinct object types in the environment (e.g., walls, doors, boxes).

    Returns:
        np.array: A flattened feature vector representing the observation.
    """
    # Extract the grid and agent's direction
    grid = observation['image']  # Assuming 'image' is the grid the agent sees
    agent_dir = observation['direction']  # 0: right, 1: down, 2: left, 3: up

    # Flatten the grid into a single vector
    grid_features = grid.flatten()

    # One-hot encode the agent's direction (0-3)
    agent_dir_onehot = np.zeros(4)
    agent_dir_onehot[agent_dir] = 1

    # Combine the flattened grid and agent's direction
    features = np.concatenate([grid_features, agent_dir_onehot])

    return features


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
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if the agent has picked up the goal object (e.g., the ball)
        if isinstance(self.carrying, Ball) and self.carrying.color == 'green':
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info