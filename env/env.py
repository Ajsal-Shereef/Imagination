import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, DIR_TO_VEC, IDX_TO_OBJECT, IDX_TO_COLOR
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

def calculate_turns_needed(agent_pos, obj_pos, agent_dir):
    """Calculate turns needed for the agent to face an object."""
    relative_position = (obj_pos[0] - agent_pos[0], obj_pos[1] - agent_pos[1])
    normalized_arr = relative_position / np.max(np.abs(relative_position))
    if np.array_equal(normalized_arr, DIR_TO_VEC[agent_dir]):  # Already aligned
        return 0
    for turns, direction in enumerate(DIR_TO_VEC, start=1):
        if np.array_equal(direction, normalized_arr):
            return min(turns, 5 - turns)  # Min turns in either direction
    return 2  # Default to two turns if diagonal (e.g., obj is far)

def calculate_probabilities(agent_position, observation, agent_direction, purple_key_position, green_ball_position):
    # Get direction vector using DIR_TO_VEC
    direction_vector = DIR_TO_VEC[agent_direction]
    
    # Calculate Manhattan distances to each ball
    distance_purple = abs(agent_position[0] - purple_key_position[0]) + abs(agent_position[1] - purple_key_position[1])
    distance_green = abs(agent_position[0] - green_ball_position[0]) + abs(agent_position[1] - green_ball_position[1])
    
    # turns_to_purple = calculate_turns_needed(agent_position, purple_key_position, agent_direction)
    # turns_to_green = calculate_turns_needed(agent_position, green_ball_position, agent_direction)
    
    # distance_purple += turns_to_purple
    # distance_green += turns_to_green
    
    # # Check if the agent is directly facing each ball
    # def is_facing(agent_pos, agent_dir_vector, obj_pos):
    #     delta_x, delta_y = obj_pos[0] - agent_pos[0], obj_pos[1] - agent_pos[1]
    #     return (delta_x, delta_y) == tuple(agent_dir_vector)
    

    # facing_purple = is_facing(agent_position, direction_vector, purple_key_position)
    # facing_green = is_facing(agent_position, direction_vector, green_ball_position)
        
    object, _ = generate_caption(observation)
    
    # # Adjust scores based on distance and facing direction
    # if "purple key" in object:
    #     score_purple = (-distance_purple) + (3 if facing_purple else 0)
    # else:
    #     score_purple = -100
    # if "green ball" in object:
    #     score_green = (-distance_green) + (3 if facing_green else 0)
    # else:
    #     score_green = -100
    
    # # Convert scores to probabilities using softmax
    # exp_scores = np.exp([score_purple, score_green])
    # probabilities = exp_scores / np.sum(exp_scores)
    if 'purple key' and 'green ball' in object:
        if distance_purple<distance_green:
            probabilities = np.array([1,0,0])
        elif distance_purple>distance_green:
            probabilities = np.array([0,1,0])
        else:
            probabilities = np.array([0,0,1])
    elif 'purple key' in object:
        probabilities = np.array([1,0,0])
    elif 'green ball' in object:
        probabilities = np.array([0,1,0])
    else:
        probabilities = np.array([0,0,1])
        
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

class MiniGridTransitionDescriber:
    def __init__(self, view_size):
        self.view_size = view_size

    def get_objects_in_view(self, view):
        """Extract objects from the agent's partial view."""
        objects_in_view = {}
        for x in range(self.view_size):
            for y in range(self.view_size):
                obj_type, color, state = view[x, y]
                obj_name = IDX_TO_OBJECT.get(obj_type, None)
                color_name = IDX_TO_COLOR.get(color, None)
                if obj_name and obj_name != "empty" and obj_name != "wall" and (x, y) != (2, 4):
                    objects_in_view[(x, y)] = (color_name, obj_name)
        return objects_in_view

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def generate_description(self, agent_prev_pos, agent_curr_pos, agent_prev_dir, agent_curr_dir, 
                             prev_view, curr_view, purple_key_pos, green_ball_pos, agent_action):
        """
        Generate a description of the agent's transition.

        Args:
        - agent_prev_pos: Tuple (x, y) of the agent's previous position.
        - agent_curr_pos: Tuple (x, y) of the agent's current position.
        - agent_prev_dir: Integer representing the agent's previous direction.
        - agent_curr_dir: Integer representing the agent's current direction.
        - prev_view: Partial view array (view_size x view_size x 3) for the previous frame.
        - curr_view: Partial view array (view_size x view_size x 3) for the current frame.
        - purple_key_pos: Tuple (x, y) of the purple key's position.
        - green_ball_pos: Tuple (x, y) of the green ball's position.
        - agent_action: Integer representing the agent's current action.

        Returns:
        - A string description of the agent's transition.
        """
        directions = ["right", "down", "left", "up"]
        agent_moved = False

        # Get objects in view before and after the transition
        curr_objects = self.get_objects_in_view(curr_view)
        if (agent_prev_pos or agent_prev_dir) is None:
            if curr_objects:
                words = []
                for pos, (color_name, obj_name) in curr_objects.items():
                    obj_words = " ".join([color_name, obj_name])
                    words.append(obj_words)
                if len(curr_objects) > 1:
                    purple_dist = self.manhattan_distance(agent_curr_pos, purple_key_pos)
                    green_dist = self.manhattan_distance(agent_curr_pos, green_ball_pos)
                    if purple_dist == green_dist:
                        # Calculate turns needed to face each ball
                        turns_to_purple = calculate_turns_needed(agent_curr_pos, purple_key_pos, agent_curr_dir)
                        turns_to_green = calculate_turns_needed(agent_curr_pos, green_ball_pos, agent_curr_dir)
                        if turns_to_purple < turns_to_green:
                            return f"the agent sees {', '.join(words)}. agent is closer to the purple key."
                        elif turns_to_green < turns_to_purple:
                            return f"the agent sees {', '.join(words)}. agent is closer to the green ball."
                        else:
                            return f"the agent sees {', '.join(words)}. agent is equidistant from both objects."
                    elif purple_dist < green_dist:
                        return f"the agent sees {', '.join(words)}. agent is closer to the purple key."
                    else:
                        return f"the agent sees {', '.join(words)}. agent is closer to the green ball."
                else:
                    return f"the agent sees {', '.join(words)}."
            else:
                return "no significant change detected."

        prev_objects = self.get_objects_in_view(prev_view)
        
        # Identify objects that disappeared from view
        disappeared_objects = [obj for pos, obj in prev_objects.items() if obj not in curr_objects.values()]

        # Identify objects that appeared in view
        appeared_objects = [obj for pos, obj in curr_objects.items() if obj not in prev_objects.values()]
        
        # Identify the common objects in both frames
        common_objects = [obj for pos, obj in curr_objects.items() if obj in prev_objects.values()]

        # Check movement description based on Manhattan distance for purple and green balls if in view
        movement_desc = ""
        for obj in common_objects:
            if obj == ('purple', 'key'):
                purple_prev_dist = self.manhattan_distance(agent_prev_pos, purple_key_pos)
                purple_curr_dist = self.manhattan_distance(agent_curr_pos, purple_key_pos)
                if purple_curr_dist < purple_prev_dist:
                    movement_desc += f"the agent moved towards to the {obj[0]} {obj[1]}. "
                    agent_moved = True
            elif obj == ('green', 'ball'):
                green_prev_dist = self.manhattan_distance(agent_prev_pos, green_ball_pos)
                green_curr_dist = self.manhattan_distance(agent_curr_pos, green_ball_pos)
                if green_curr_dist < green_prev_dist:
                    movement_desc += f"the agent moved towards to the {obj[0]} {obj[1]}. "
                    agent_moved = True
                    
        if len(common_objects)>1 and agent_moved:
            movement_desc = ""
            if green_curr_dist>purple_curr_dist:
                movement_desc += f"the agent moved towards to the purple key. "
            elif green_curr_dist<purple_curr_dist:
                movement_desc += f"the agent moved towards to the green ball. "
            elif green_curr_dist==purple_curr_dist:
                turns_to_purple = calculate_turns_needed(agent_curr_pos, purple_key_pos, agent_curr_dir)
                turns_to_green = calculate_turns_needed(agent_curr_pos, green_ball_pos, agent_curr_dir)
                if turns_to_purple < turns_to_green:
                    movement_desc += "the agent moved towards to the purple key. "
                elif turns_to_green < turns_to_purple:
                    movement_desc += "the agent moved towards to the green ball. "
                else:
                    movement_desc += "the agent moved equally towards both objects "
                    
        # Describe objects that disappeared from the view
        disappearance_desc = ""
        words = []
        if disappeared_objects:
            for color_name, obj_name in disappeared_objects:
                if agent_action == 3:
                    disappearance_desc += f"the agent picked the {color_name} {obj_name}. "
                # else:
                #     obj_words = " ".join([color_name, obj_name])
                #     words.append(obj_words)
            # if words:
            #     disappearance_desc += f"the agent ignored {', '.join(words)}. "
                
        # Describe objects that appeared in the view
        # appearance_desc = ""
        # words = []
        # if appeared_objects:
        #     for color_name, obj_name in appeared_objects:
        #         obj_words = " ".join([color_name, obj_name])
        #         words.append(obj_words)
        #     if len(curr_objects) > 1:
        #         purple_dist = self.manhattan_distance(agent_curr_pos, purple_key_pos)
        #         green_dist = self.manhattan_distance(agent_curr_pos, green_ball_pos)
        #         if purple_dist < green_dist:
        #             appearance_desc += f"the {', '.join(words)} appeared in the current view, agent is closer to the purple key."
        #         elif purple_dist > green_dist:
        #             appearance_desc += f"the {', '.join(words)} appeared in the current view, agent is closer to the green ball."
        #         else:
        #             turns_to_purple = calculate_turns_needed(agent_curr_pos, purple_key_pos, agent_curr_dir)
        #             turns_to_green = calculate_turns_needed(agent_curr_pos, green_ball_pos, agent_curr_dir)
        #             if turns_to_purple < turns_to_green:
        #                 appearance_desc += f"the {', '.join(words)} appeared in the current view, agent is closer to the purple key."
        #             elif turns_to_green < turns_to_purple:
        #                 appearance_desc += f"the {', '.join(words)} appeared in the current view, agent is closer to the green ball."
        #             else:
        #                 appearance_desc += f"the {', '.join(words)} appeared in the current view, agent is equidistant from both objects."
        #     else:
        #         appearance_desc += f"the {', '.join(words)} appeared in the current view."

        # Combine all parts into a single description
        # return (movement_desc + appearance_desc + disappearance_desc).strip()  if (
        #     movement_desc or appearance_desc or disappearance_desc)  else "no significant change detected"
        return (movement_desc + disappearance_desc).strip()  if (
            movement_desc or disappearance_desc)  else "no significant change detected"


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
        # self.observation_space = spaces.Box(low=0, high=1, shape=(79,), dtype=np.float32)

    @staticmethod
    def _gen_mission():
        return "Pick the green ball"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # # # # Place a ball square in the bottom-right corner
        # self.grid.set(4, 2, Ball(color='green'))
        # self.green_ball_loc = (4,2)
        
        # # # # Place a ball square in the bottom-right corner
        # self.grid.set(2, 4, Ball(color='red'))
        # self.purple_key_loc = (2,4)
        
        # Place one green ball at a random position
        self.green_ball_loc = self.place_obj(Ball('green', can_overlap=False), max_tries=100)

        # Place one purple key at a random position
        self.purple_key_loc = self.place_obj(Key('purple', can_overlap=False), max_tries=100)

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
    

    # def reset(self, seed):
    #     self.obs, info = super().reset(seed)
    #     # obs = preprocess_observation(self.obs)
    #     return self.obs, info
        
    def step(self, action):
        self.obs, reward, terminated, truncated, info = super().step(action)
        # obs = preprocess_observation(self.obs)

        # Check if the agent has picked up the goal object (e.g., the ball)
        if isinstance(self.carrying, Ball) and self.carrying.color == 'green':
            reward = self._reward()
            terminated = True
        elif isinstance(self.carrying, Key) and self.carrying.color == 'purple':
            terminated = True
        return self.obs, reward, terminated, truncated, info