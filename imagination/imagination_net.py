import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from architectures.mlp import MLP
from architectures.Layers import *
from architectures.film import FiLMLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

class ImaginationNet(nn.Module):
    def __init__(self,
                 env,
                 config,
                 num_goals,
                 goal_vector,
                 agent,
                 sentence_encoder
                ):
        """
        Imagination network that accepts a state and outputs N imagined states.
    
        Args:
            state_dim (int): Dimensionality of the input state.
            N (int): Number of imagined states (corresponding to the number of Gaussians).
        """
        super(ImaginationNet, self).__init__()
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        hidden_layers = config.Network.hidden_layers
        self.sentence_encoder = sentence_encoder
        self.film_layer = FiLMLayer(self.input_dim, goal_vector.shape[1])
        self.imagination_net = MLP(input_size = self.input_dim,
                                   output_size = num_goals*self.input_dim,
                                   hidden_sizes = hidden_layers,
                                   output_activation = F.relu,
                                   dropout_prob = 0.20)
        self.goal_vector = goal_vector
        self.agent = agent
        self.agent.eval()
        self.num_goals = num_goals
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        #Preventing agent to take random action.
        if config.General.agent == 'dqn':
            self.agent.set_episode_step()
    
    # Loss function: 
    def compute_loss(self, state, imagined_states, m=1.0):
        """
        Computes the total loss for the imagined states. The loss consists of:
        1. Cross-entropy between VAE output and one-hot vector for Gaussian component.
        2. L2 norm between imagined states and input state.
        3. Cross-entropy between SAC policy for imagined states and the action taken.

        Args:
            state (Tensor): Input state of shape [batch_size, state_dim].
            m (float): Coefficient for the policy loss term.
        
        Returns:
            Tensor: Computed loss.
        """
        # q1_original = self.sac_agent.critic1(state)
        # q2_original = self.sac_agent.critic2(state)
        # q_values_original = torch.min(q1_original,q2_original)
        total_loss = 0.0
        goal_loss = 0.0
        policy_loss = 0.0
        proximity_loss = 0.0

        # Loop over each imagined state
        for i in range(self.num_goals):
            imagined_state = imagined_states[:, i, :]
            # 1. Cross-entropy between VAE output and one-hot vector for Gaussian component i
            with torch.no_grad():
                state_array = np.round(imagined_state.detach().cpu().numpy())
                state_array_grid = state_array[:,:-4].reshape((state_array.shape[0], 5,5,3))
                imagined_state_captions =  [self.env.generate_caption(obs) for obs in state_array_grid]
                imagined_state_encode = self.sentence_encoder.encode(imagined_state_captions, convert_to_tensor=True, device=device, show_progress_bar=False)
                imagined_state_encode.requires_grad_()
            cosine_sim = torch.mean(F.cosine_similarity(imagined_state_encode, self.goal_vector[i,:].unsqueeze(0), dim=1))

            # 2. L2 distance between imagined state and input state
            l2_loss = self.mse_loss(imagined_state, state)

            # 3. Cross-entropy between SAC policy and one-hot vector of action taken
            with torch.no_grad():
                original_state_action = self.agent.get_action(state)
                imagined_state_action = self.agent.get_action(imagined_state)
                
            # Create a one-hot tensor
            one_hot_action = F.one_hot(original_state_action, num_classes=self.env.action_space.n).float()
            one_hot_imagined_action = F.one_hot(imagined_state_action, num_classes=self.env.action_space.n).float()
            one_hot_action.requires_grad_()
            one_hot_imagined_action.requires_grad_()
                # q1 = self.sac_agent.critic1(imagined_state)
                # q2 = self.sac_agent.critic2(imagined_state)
                # q_values = torch.min(q1,q2)
            # one_hot_action = torch.zeros_like(policy_imagined)
            # one_hot_action.scatter_(1, sac_action.unsqueeze(1), 1)  # One-hot for action taken

            ce_loss_policy = self.mse_loss(one_hot_action, one_hot_imagined_action)

            # Total loss for this imagined state
            total_loss += -cosine_sim + 0.8*l2_loss + 0.8*ce_loss_policy * m
            
            goal_loss += cosine_sim
            policy_loss += ce_loss_policy
            proximity_loss += l2_loss
            
        # Return the average loss across all N imagined states
        return total_loss, goal_loss, policy_loss, proximity_loss 
        
    def forward(self, state, caption):
        """
        Forward pass through the network. Given a state, outputs N imagined states.
        
        Args:
            state (Tensor): Input state of shape [batch_size, state_dim].
        Returns:
            Tensor: Imagined states of shape [batch_size, N, state_dim].
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float().unsqueeze(0).to(device)
        conditioned_vector = self.film_layer(state, caption)
        imagined_state = self.imagination_net(conditioned_vector).view(-1, self.num_goals, self.input_dim)
        return imagined_state
    
    def save(self, path):
        """
        Save the model's state dictionary.

        Args:
            path (str): File path to save the model.
        """
        params = {
        "film_layer": self.film_layer.state_dict(),
        "imagination_net": self.imagination_net.state_dict()
        }
        torch.save(params, path)
        print(f"[INFO] Model saved to {path}")

    def load(self, path):
        """
        Load the model's state dictionary.

        Args:
            path (str): File path from which to load the model.
        """
        params = torch.load(path, map_location=device)
        self.film_layer.load_state_dict(params["film_layer"])
        self.imagination_net.load_state_dict(params["imagination_net"])
        print(f"[INFO] Model loaded from {path}")