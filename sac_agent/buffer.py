import os
import pickle
import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class TrajectoryReplyBuffer:
    """Buffer to store the trajectory data"""
    def __init__(self, max_episode_len, feature_dim, buffer_size, encoded_dim):
        self.max_episode_len = max_episode_len
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        self.encoded_dim = encoded_dim
        self.next_spot_to_add = 0
        self.buffer_is_full = False
        
        self.states_buffer = np.zeros(shape=(self.buffer_size, self.max_episode_len+1, self.feature_dim), dtype=np.float32)
        self.caption_buffer = np.zeros(shape=(self.buffer_size, self.max_episode_len+1, self.encoded_dim), dtype=np.float32)
        
    def dump_buffer_data(self, dump_dir):
        
        data_path = os.path.join(dump_dir, 'caption')
        np.save(data_path, self.caption_buffer)
        print(f"[INFO] states data with len {len(self.caption_buffer)} saved to {data_path}")
        
        # data_path = os.path.join(dump_dir, 'states')
        np.save(dump_dir + '/states', self.states_buffer)
        print(f"[INFO] states data with len {len(self.states_buffer)} saved to {data_path}")
            
            
    def add(self, state, caption):
        traj_length = len(state)
        self.next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + 1
        if self.next_spot_to_add >= self.buffer_size:
            self.buffer_is_full = True
        self.states_buffer[self.next_ind, :traj_length] = state
        self.states_buffer[self.next_ind, traj_length:] = 0
        self.caption_buffer[self.next_ind, :traj_length] = caption
        self.caption_buffer[self.next_ind, traj_length:] = 0
        
    def sample(self, batch_size):
        indices = np.random.choice(range(self.buffer_size), size = batch_size)
        return (self.states_buffer[indices, :, :], self.caption_buffer[indices, :, :], indices)