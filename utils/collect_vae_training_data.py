import cv2
import argparse
import os
import pickle
import random
import numpy as np
from collections import deque

import minigrid
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import sys
sys.path.append('.')

from env.env import calculate_probabilities, generate_caption

encoder = "all-MiniLM-L12-v2"

# =============================
# 1. Define the Q-Network Class
# =============================

class QNetwork(nn.Module):
    """
    Example Q-Network. 
    **Important**: Ensure that the architecture matches the one used during training.
    Modify this class according to your pretrained Q-network's architecture.
    """
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # Simple MLP; replace with your architecture
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        """
        Forward pass of the Q-network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# =============================
# 2. Argument Parsing
# =============================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect transition data from MiniGrid environment.")
    parser.add_argument(
        '--env',
        type=str,
        default = "SimplePickup",
        help='Name of the MiniGrid environment (e.g., MiniGrid-Empty-5x5-v0)'
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--random',
        action='store_true',
        default=True,
        help='Use a random policy.'
    )
    group.add_argument(
        '--q-network',
        type=str,
        help='Path to the pretrained Q-network.'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1500,
        help='Number of episodes to run for data collection (default: 100)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum steps per episode (default: 100)'
    )
    
    args = parser.parse_args()
    return args

# =============================
# 3. Helper Functions
# =============================

# def preprocess_observation(obs):
#     """
#     Preprocess the observation to a suitable format for the Q-network.
    
#     Args:
#         obs (dict): Observation from the MiniGrid environment.
    
#     Returns:
#         torch.Tensor: Preprocessed observation tensor.
#     """
#     # Example preprocessing:
#     # Flatten the 'image' field and concatenate with 'direction' and other scalar observations.
#     # Modify this function based on how your Q-network expects the input.
    
#     # Flatten the image
#     img = obs['image']  # Shape: (7, 7, 3)
#     img_flat = img.flatten()  # Shape: (147,)
    
#     # Get direction
#     direction = torch.tensor([obs['direction']], dtype=torch.float32)  # Shape: (1,)
    
#     # Get mission as one-hot or encoded
#     mission = obs['mission']  # String
#     # Simple encoding: use a fixed-size one-hot vector or embed
#     # Here, we'll use a random embedding for demonstration; replace with actual encoding
#     # For simplicity, ignore 'mission' or use a placeholder
#     # Alternatively, use a separate embedding or ignore it
#     # Here, we will ignore it
    
#     # Combine features
#     features = torch.tensor(img_flat, dtype=torch.float32)
#     features = torch.cat((features, direction), dim=0)  # Shape: (148,)
    
    # return features

def select_action_random(action_space):
    """
    Select a random action.
    
    Args:
        action_space: The action space of the environment.
    
    Returns:
        int: Selected action.
    """
    return action_space.sample()

def select_action_q_network(q_network, state, device):
    """
    Select the best action based on Q-network predictions.
    
    Args:
        q_network (nn.Module): The pretrained Q-network.
        state (torch.Tensor): Preprocessed state tensor.
        device (torch.device): Device to run the Q-network on.
    
    Returns:
        int: Selected action.
    """
    q_network.eval()
    with torch.no_grad():
        state = state.unsqueeze(0).to(device)  # Add batch dimension
        q_values = q_network(state)  # Shape: (1, num_actions)
        action = torch.argmax(q_values, dim=1).item()
    return action

# =============================
# 4. Data Collection
# =============================

def collect_data(env, use_random, episodes, max_steps, device, q_network_path=None):
    """
    Collect transition data from the specified environment using the chosen policy.
    
    Args:
        env_name (str): Name of the MiniGrid environment.
        use_random (bool): Whether to use a random policy.
        q_network_path (str or None): Path to the pretrained Q-network.
        episodes (int): Number of episodes to run.
        max_steps (int): Maximum steps per episode.
        device (torch.device): Device to run the Q-network on.
    
    Returns:
        list: Collected transitions as dictionaries.
    """
    data = []
    captions = []
    class_prob = []
    
    # Load the sentecebert model to get the embedding of the goals from the LLM
    sentencebert = SentenceTransformer(encoder)

    # Initialize Q-network if not using random policy
    if not use_random:
        # Define the input and output dimensions based on preprocessing
        # Modify input_dim and output_dim according to your Q-network
        # Example assumes input_dim=148 and output_dim=env.action_space.n
        input_dim = 148  # Adjust based on preprocess_observation
        output_dim = env.action_space.n
        q_network = QNetwork(input_dim, output_dim).to(device)
        
        # Load the Q-network weights
        if not os.path.isfile(q_network_path):
            raise FileNotFoundError(f"Q-network file not found at: {q_network_path}")
        
        q_network.load_state_dict(torch.load(q_network_path, map_location=device))
        q_network.eval()
        print(f"Loaded Q-network from {q_network_path}")
    else:
        q_network = None
        print("Using random policy.")
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        obj, caption = generate_caption(env.get_unprocesed_obs()['image'])
        
        # frame = env.get_frame()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("previous_frame.png",frame)
        
        prob = calculate_probabilities(env.agent_pos, 
                                       env.get_unprocesed_obs()['image'], 
                                       env.get_unprocesed_obs()['direction'], 
                                       env.red_ball_loc, 
                                       env.green_ball_loc)
        class_prob.append(prob)
        caption_encoding = sentencebert.encode(caption, convert_to_tensor=True, device=device)
        captions.append(caption_encoding)
        #caption = state_captioner.generate_caption(state)
        # state = preprocess_observation(state)
        data.append(state)
        
        done = False
        step = 0
        while not done:
            # state = preprocess_observation(obs)  # Preprocess observation
            if use_random:
                action = select_action_random(env.action_space)
            else:
                action = select_action_q_network(q_network, state, device)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # frame = env.get_frame()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("frame.png", frame)
            
            prob = calculate_probabilities(env.agent_pos, 
                                           env.get_unprocesed_obs()['image'], 
                                           env.get_unprocesed_obs()['direction'], 
                                           env.red_ball_loc, 
                                           env.green_ball_loc)
            class_prob.append(prob)
            done = terminated + truncated
            obj, caption = generate_caption(env.get_unprocesed_obs()['image'])
            caption_encoding = sentencebert.encode(caption, convert_to_tensor=True, device=device)
            captions.append(caption_encoding)
            # next_state = preprocess_observation(next_state)
            # Store the transition
            data.append(next_state)
            
            state = next_state
            step += 1
        
        print(f"Episode {episode}/{episodes} finished after {step} steps.")
    
    env.close()
    return data, captions, class_prob

# =============================
# 5. Main Function
# =============================

def main():
    args = parse_arguments()
    
    env_name = args.env
    use_random = args.random
    q_network_path = args.q_network
    episodes = args.episodes
    max_steps = args.max_steps
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data directory if it doesn't exist
    data_dir = f'data/{env_name}'
    os.makedirs(data_dir, exist_ok=True)
    
    if env_name == "SimplePickup":
        from env.env import SimplePickup #TransitionCaptioner
        env = SimplePickup(max_steps=max_steps, agent_view_size=5, size=7)
        # transition_captioner = 
    
    # Collect data
    data, captions, class_prob = collect_data(
        env=env,
        use_random=use_random,
        q_network_path=q_network_path,
        episodes=episodes,
        max_steps=max_steps,
        device=device
    )
    
    # Save data
    data_path = os.path.join(data_dir, 'data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Collected {len(data)} transitions and saved to {data_path}")
    
    data_path = os.path.join(data_dir, 'captions.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(captions, f)
    print(f"Collected {len(captions)} captions and saved to {data_path}")
    
    data_path = os.path.join(data_dir, 'class_prob.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(class_prob, f)
    print(f"Collected {len(captions)} class prob and saved to {data_path}")
# =============================
# 6. Entry Point
# =============================

if __name__ == "__main__":
    main()
