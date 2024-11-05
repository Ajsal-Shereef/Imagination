import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, DIR_TO_VEC
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

def get_object_name(obj_type, obj_color, obj_state):
        """Returns a string description of an object based on its type, color, and state."""
        #Invert the dictionary
        IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
        IDX_TO_OBJECTS = {v: k for k, v in OBJECT_TO_IDX.items()}
        IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
        # Map indices to names
        color_name = IDX_TO_COLOR.get(int(obj_color), "unknown color)")
        object_name = IDX_TO_OBJECTS.get(int(obj_type), "unknown object")
        state_name = IDX_TO_STATE.get(int(obj_state), "")
        # Construct a descriptive name
        return f"{color_name} {object_name}".strip()

def generate_caption(observation):
    """
    Generate a natural language caption for the MiniGrid observation.
    Parameters:
        observation (dict): The observation dictionary with keys 'image', 'direction', 'mission'.
    Returns:
        str: A caption describing the state.
    """
    # Extract relevant data from observation
    image = observation
    view_size = image.shape[0]
    agent_position = (view_size // 2, view_size - 1)
    # Initialize lists to store object descriptions and distances
    descriptions = []
    distances = []
    # Iterate over the field of view and collect descriptions for each visible object
    for i in range(view_size):
        for j in range(view_size):
            # Extract object type, color, and state from the image
            obj_type, obj_color, obj_state = image[i, j]
            # Skip "wall" and "empty" objects based on their indices
            if obj_type in {OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['empty']}:
                continue
            
            # Generate a natural language description of the object
            obj_description = get_object_name(obj_type, obj_color, obj_state)
            # Calculate the Euclidean distance from the agent to this object
            distance = np.sqrt((i - agent_position[0])**2 + (j - agent_position[1])**2)
            # Store the object description and distance
            descriptions.append(obj_description)
            distances.append((distance, obj_description))
    # Generate the final caption based on the collected descriptions
    if not descriptions:
        # No objects detected in the view
        return {}, "You see nothing."
    # Sort the objects by proximity and identify the closest one
    distances.sort()
    min_distance = distances[0][0]
    closest_objects = [desc for dist, desc in distances if dist == min_distance]
    # Join all objects descriptions
    objects_seen = ", ".join(descriptions)
    # Create the caption
    # Create the caption mentioning all closest objects
    if len(closest_objects) == 1:
        caption = f"You see {objects_seen} and you are close to {closest_objects[0]}."
    else:
        closest_desc = " and ".join(closest_objects)
        caption = f"You see {objects_seen} and you are close to {closest_desc}."
    return objects_seen, caption

def calculate_probabilities(agent_position, observation, agent_direction, red_ball_position, green_ball_position):
    # Get direction vector using DIR_TO_VEC
    direction_vector = DIR_TO_VEC[agent_direction]
    
    # Calculate Manhattan distances to each ball
    distance_red = abs(agent_position[0] - red_ball_position[0]) + abs(agent_position[1] - red_ball_position[1])
    distance_green = abs(agent_position[0] - green_ball_position[0]) + abs(agent_position[1] - green_ball_position[1])
    
    # Check if the agent is directly facing each ball
    def is_facing(agent_pos, agent_dir_vector, obj_pos):
        delta_x, delta_y = obj_pos[0] - agent_pos[0], obj_pos[1] - agent_pos[1]
        return (delta_x, delta_y) == tuple(agent_dir_vector)
    

    facing_red = is_facing(agent_position, direction_vector, red_ball_position)
    facing_green = is_facing(agent_position, direction_vector, green_ball_position)
        
    object, _ = generate_caption(observation)
    
    # Adjust scores based on distance and facing direction
    if "red ball" in object:
        score_red = (-distance_red) + (5 if facing_red else 0)
    else:
        score_red = -10
    if "green ball" in object:
        score_green = (-distance_green) + (5 if facing_green else 0)
    else:
        score_green = -10
    
    # Convert scores to probabilities using softmax
    exp_scores = np.exp([score_red, score_green])
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities


# def preprocess_observation(obs):
#     """
#     Preprocesses the MiniGrid observation using one-hot encoding, including the agent's direction.
    
#     Args:
#         obs: Observation from the environment, a dictionary with keys `image`, `direction`, and `mission`.
#     Returns:
#         processed_obs: Preprocessed observation with one-hot encoded features for objects, colors, states, and agent direction, flattened into a 1D vector.
#     """
#     # Extract the grid (partially observable view) and the agent's direction
#     grid_obs = obs['image']  # Shape: (grid_height, grid_width, 3)
#     agent_direction = obs['direction']  # Scalar value for direction (0 to 3)

#     # Get dimensions for the grid (height, width) and the number of visible cells
#     height, width, _ = grid_obs.shape
#     num_cells = height * width
    
#     # Precompute the sizes for one-hot encoding based on the constants
#     num_objects = len(OBJECT_TO_IDX)  # Total number of objects
#     num_colors = len(COLOR_TO_IDX)    # Total number of colors
#     num_states = len(STATE_TO_IDX)    # Total number of states (open, closed, etc.)
    
#     # Create an empty list to store the encoded features
#     processed_obs = []

#     # Iterate over each cell in the grid (partially observable view)
#     for i in range(height):
#         for j in range(width):
#             # Each cell has an object, color, and state
#             obj_type = grid_obs[i, j, 0]
#             obj_color = grid_obs[i, j, 1]
#             obj_state = grid_obs[i, j, 2]
            
#             # One-hot encode the object type, color, and state
#             obj_one_hot = get_one_hot_encoding(obj_type, num_objects)
#             color_one_hot = get_one_hot_encoding(obj_color, num_colors)
#             state_one_hot = get_one_hot_encoding(obj_state, num_states)
            
#             # Concatenate the one-hot encoded features for this cell
#             encoded_cell = np.concatenate([obj_one_hot, color_one_hot, state_one_hot])
            
#             # Append to the list of processed observations
#             processed_obs.append(encoded_cell)
    
#     # Convert the list to a numpy array and reshape it to match the flattened view of the grid
#     processed_obs = np.array(processed_obs).reshape((num_cells, -1))
    
#     # One-hot encode the agent's direction and concatenate it to the processed observation
#     direction_one_hot = get_one_hot_encoding(agent_direction, NUM_DIRECTIONS)
    
#     # Concatenate direction encoding with the processed grid observation
#     processed_obs = np.concatenate([processed_obs.flatten(), direction_one_hot])

#     return processed_obs

def preprocess_observation(obs):
    grid_obs = obs['image']  # Shape: (grid_height, grid_width, 3)
    agent_direction = obs['direction']  # Scalar value for direction (0 to 3)
    direction_one_hot = get_one_hot_encoding(agent_direction, NUM_DIRECTIONS)
    return np.concatenate([grid_obs.flatten(), direction_one_hot])

def one_hot_to_index(one_hot_vec):
    """Helper function to convert a one-hot encoded vector back to the index."""
    return np.argmax(one_hot_vec)

def reverse_preprocess_observation(processed_obs, height, width):
    """
    Reverses the preprocessing of a MiniGrid observation, returning the original observation format.
    
    Args:
        processed_obs: Preprocessed observation (1D vector).
        height: The height of the observation grid (view_size).
        width: The width of the observation grid (view_size).
        
    Returns:
        original_obs: Dictionary with 'image' (grid of [object_type, color, state]) and 'direction'.
    """
    # Define constants for sizes
    num_objects = len(OBJECT_TO_IDX)
    num_colors = len(COLOR_TO_IDX)
    num_states = len(STATE_TO_IDX)
    num_directions = NUM_DIRECTIONS  # The total number of possible directions (4 in MiniGrid)

    # Compute the size of one cell's encoding (object, color, state concatenated)
    cell_encoding_size = num_objects + num_colors + num_states

    # Total number of cells in the grid
    num_cells = height * width

    # Split the flattened processed_obs into grid part and direction part
    grid_part = processed_obs[:-num_directions]
    direction_part = processed_obs[-num_directions:]

    # Reshape grid part back into (num_cells, cell_encoding_size)
    grid_part = grid_part.reshape((num_cells, cell_encoding_size))

    # Initialize the original observation image
    original_image = np.zeros((height, width, 3), dtype=int)

    # Iterate over each cell in the grid and decode the object, color, and state
    for cell_idx in range(num_cells):
        i = cell_idx // width  # Row index
        j = cell_idx % width   # Column index

        # Extract the one-hot encoded parts for object, color, and state
        obj_one_hot = grid_part[cell_idx][:num_objects]
        color_one_hot = grid_part[cell_idx][num_objects:num_objects + num_colors]
        state_one_hot = grid_part[cell_idx][num_objects + num_colors:]

        # Convert back to indices
        obj_type = one_hot_to_index(obj_one_hot)
        obj_color = one_hot_to_index(color_one_hot)
        obj_state = one_hot_to_index(state_one_hot)

        # Reconstruct the original observation grid cell
        original_image[i, j, 0] = obj_type
        original_image[i, j, 1] = obj_color
        original_image[i, j, 2] = obj_state

    # Decode the agent's direction from the one-hot encoded direction part
    agent_direction = one_hot_to_index(direction_part)

    # Reconstruct the original observation dictionary
    original_obs = {
        'image': original_image,
        'direction': agent_direction
    }

    return original_obs


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

class TransitionCaptioner:
    def __init__(self, view_size):
        self.view_size = view_size
        # Reverse dictionaries for decoding
        self.idx_to_object = {v: k for k, v in OBJECT_TO_IDX.items()}
        self.idx_to_color = {v: k for k, v in COLOR_TO_IDX.items()}
        self.idx_to_state = {v: k for k, v in STATE_TO_IDX.items()}
        self.directions = {
            0: "right",
            1: "down",
            2: "left",
            3: "up"
        }

    def decode_observation(self, obs_image):
        """Decode an observation from integers to readable names."""
        decoded = []
        for i in range(self.view_size):
            row = []
            for j in range(self.view_size):
                obj_idx, color_idx, state_idx = obs_image[i, j]
                obj = self.idx_to_object.get(obj_idx, "empty")
                color = self.idx_to_color.get(color_idx, "unknown")
                state = self.idx_to_state.get(state_idx, "unknown")
                row.append((obj, color, state))
            decoded.append(row)
        return decoded

    def describe_change(self, old, new, direction):
        """Generate a natural language description of the change between two observations."""
        changes = []
        for i in range(self.view_size):
            for j in range(self.view_size):
                old_obj, old_color, old_state = old[i][j]
                new_obj, new_color, new_state = new[i][j]

                # Skip if object is "wall" or "empty" or if there was no change
                if old_obj in ["wall", "empty"] and new_obj in ["wall", "empty"]:
                    continue
                if (old_obj, old_color, old_state) == (new_obj, new_color, new_state):
                    continue

                direction_desc = f"{self.directions[direction]}"
                
                # If the object disappeared and isn't "wall" or "empty", assume it was picked up
                if old_obj != "empty" and new_obj == "empty":
                    changes.append(f"The agent picked up the {old_color} {old_obj} at position ({i}, {j}), relative to the agent's {direction_desc}.")
                elif old_obj == "empty" and new_obj != "empty":
                    changes.append(f"A {new_color} {new_obj} appeared at position ({i}, {j}), relative to the agent's {direction_desc}.")
                elif old_obj != new_obj:
                    changes.append(f"The object at position ({i}, {j}), relative to the agent's {direction_desc}, changed from {old_color} {old_obj} to {new_color} {new_obj}.")
                elif old_state != new_state:
                    changes.append(f"The {new_color} {new_obj} at position ({i}, {j}), relative to the agent's {direction_desc}, changed its state from {old_state} to {new_state}.")
        
        if changes:
            return " ".join(changes)
        else:
            return "No notable changes."

    def get_transition_caption(self, old_obs, new_obs):
        """Generates a caption for the transition between two consecutive observations."""
        old_image, new_image = old_obs['image'], new_obs['image']
        old_decoded = self.decode_observation(old_image)
        new_decoded = self.decode_observation(new_image)
        direction = new_obs['direction']  # Use the direction from the new observation
        return self.describe_change(old_decoded, new_decoded, direction)


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
        size=7,
        render_mode = None,
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
            render_mode = render_mode,
            **kwargs,
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(79,), dtype=np.float32)

    @staticmethod
    def _gen_mission():
        return "Pick the green ball"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # # Place a ball square in the bottom-right corner
        self.put_obj(Ball(color='green'), width-2, height-2)
        
        # # Place a ball square in the bottom-right corner
        self.put_obj(Ball(color='red'), 1, height-2)
        
        # Place one green ball at a random position
        # self.place_obj(Ball('green'), max_tries=100)

        # Place one red ball at a random position
        # self.place_obj(Ball('red'), max_tries=100)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Pick the green ball"
    
    # def describe_objects(self, objects):
    #     if not objects:
    #         return "You see nothing."
    #     return "You see " + ", ".join(objects) + "."
        
    # def generate_caption(self, obs):
    #     """
    #     Extracts the objects present in the partial observation, excluding empty tiles and walls.

    #     Args:
    #         observation (numpy array): Observation from the MiniGrid environment, shape (view_size, view_size, 3).

    #     Returns:
    #         Set: A set containing strings that describe the objects with their colors.
    #     """
    #     objects_seen = set()

    #     # Create reverse dictionaries to map indices back to object and color names
    #     IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
    #     IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

    #     # Iterate through the observation grid (view_size x view_size)
    #     for i in range(obs.shape[0]):
    #         for j in range(obs.shape[1]):
    #             # Extract object type, color, and state from the last dimension
    #             obj_type_code = obs[i, j, 0]
    #             obj_color_code = obs[i, j, 1]
    #             obj_state = obs[i, j, 2]  # State is currently unused but available

    #             # Map the codes to their respective descriptions
    #             obj_type = IDX_TO_OBJECT.get(obj_type_code, 'unknown')
    #             obj_color = IDX_TO_COLOR.get(obj_color_code, 'unknown')

    #             # Filter out 'empty' and 'wall' objects
    #             if obj_type not in ['empty', 'wall']:
    #                 object_description = f"{obj_color} {obj_type}"
    #                 objects_seen.add(object_description)
    #     caption = self.describe_objects(objects_seen)
    #     return objects_seen, caption
    
    # def generate_caption(self, observation):
    #     view_size = observation['image'].shape[0]
    #     image = observation['image']
    #     agent_direction = observation['direction']
    #     mission = observation['mission']
        
    #     # Define direction mapping for descriptive text
    #     direction_map = {0: "front", 1: "right", 2: "behind", 3: "left"}
        
    #     # Agent's fixed position in its field of view
    #     agent_pos = np.array([view_size - 1, view_size // 2])
    
    #     # Track objects and their positions
    #     objects = []
    #     distances = []
    #     directions = []
        
    #     # Iterate over each cell in the field of view
    #     for i in range(view_size):
    #         for j in range(view_size):
    #             obj_type, obj_color, obj_state = image[i, j]
                
    #             # Skip walls and empty spaces
    #             if obj_type in {OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['empty']}:
    #                 continue
                
    #             # Calculate Euclidean distance from agent's position to object
    #             obj_pos = np.array([i, j])
    #             distance = np.linalg.norm(obj_pos - agent_pos)
                
    #             # Determine the relative direction based on agent's orientation
    #             relative_position = obj_pos - agent_pos
    #             obj_direction = "nearby"
    #             for dir_key, dir_vector in enumerate(DIR_TO_VEC):
    #                 if np.array_equal(np.sign(relative_position), dir_vector):
    #                     obj_direction = direction_map[(agent_direction + dir_key) % 4]
    #                     break

    #             # Get object and color names
    #             obj_name = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(obj_type)]
    #             color_name = COLOR_NAMES[obj_color]
                
    #             # Record object details
    #             obj_description = f"{color_name} {obj_name}"
    #             objects.append((obj_description, obj_direction))
    #             distances.append(distance)
    #             directions.append(obj_direction)
        
    #     # If no objects were observed
    #     if not objects:
    #         return "You see nothing."
        
    #     # Determine the closest objects (can be more than one if distances match)
    #     min_distance = min(distances)
    #     closest_objects = [(desc, dir) for (desc, dir), dist in zip(objects, distances) if dist == min_distance]
        
    #     # Formulate the caption
    #     object_list = ", ".join([f"{desc} to your {dir}" for desc, dir in objects])
    #     closest_desc = " and ".join([f"{desc} to your {dir}" for desc, dir in closest_objects])
        
    #     caption = f"You see {object_list}. You are closest to {closest_desc}."
    #     return caption
    
    def get_unprocesed_obs(self):
        return self.obs
        
    def reset(self):
        self.obs, info = super().reset()
        obs = preprocess_observation(self.obs)
        return obs, info
        
    def step(self, action):
        self.obs, reward, terminated, truncated, info = super().step(action)
        obs = preprocess_observation(self.obs)

        # Check if the agent has picked up the goal object (e.g., the ball)
        if isinstance(self.carrying, Ball) and self.carrying.color == 'green':
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info