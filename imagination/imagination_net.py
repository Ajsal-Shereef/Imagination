import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from architectures.mlp import MLP, Linear
from architectures.Layers import *
from architectures.film import FiLMLayer
from env.env import MiniGridTransitionDescriber
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

class ImaginationNet(nn.Module):
    def __init__(self,
                 env,
                 config,
                 num_goals,
                 agent,
                 goals,
                 approx_embed_model,
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
        self.config = config
        self.input_dim = env.observation_space.shape[0]
        latent_dim = sentence_encoder.get_sentence_embedding_dimension()
        self.sentece_encoder = sentence_encoder
        self.approx_embed_model = approx_embed_model
        self.goals = goals
        hidden_layers = config.Imagination_Network.hidden_layers
        self.encoder = MLP(input_size = self.input_dim,
                           output_size = config.Imagination_Network.output_dim,
                           hidden_sizes = hidden_layers,
                           output_activation = F.relu,
                           dropout_prob = 0.0)
        self.film_layer = FiLMLayer(config.Imagination_Network.output_dim, latent_dim)
        self.fc1 = Linear(in_dim = latent_dim, 
                          out_dim = self.input_dim)
        self.fc2 = Linear(in_dim = latent_dim, 
                          out_dim = self.input_dim)
        self.agent = agent
        self.num_goals = num_goals
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        #Preventing agent to take random action.
        # if config.General.agent == 'dqn':
        #     self.agent.set_episode_step()

    def compute_loss(self, state, imagined_states):
        """
        Computes the total loss for the imagined states. The loss consists of:
        1. Negative log-likelihood between VAE class output and target class distribution.
        2. L2 norm (proximity loss) between imagined states and input state.
        3. MSE between SAC policy for imagined states and the action taken.

        Args:
            state (Tensor): Input state of shape [batch_size, state_dim].
            imagined_states (Tensor): Imagined states of shape [batch_size, num_goals, state_dim].

        Returns:
            Tuple[Tensor]: Total loss and individual component losses: class_loss, policy_loss, proximity_loss.
        """
        total_loss = 0.0
        class_loss = 0.0
        policy_loss = 0.0
        proximity_loss = 0.0
        b,t,f = state.shape
        # Forward pass through SAC agent for original state
        agent_action = self.agent.actor_local(state.view(b*t,-1)).to(state.device)  # Get agent's action for original state
        # Loss for each imagined state
        for i in range(self.num_goals):
            imagined_state = imagined_states[:, i, :]
            # 1. Proximity loss between imagined state and original state
            prox_loss = F.mse_loss(imagined_state, state)

            # 2. Action consistency loss with SAC policy
            imagined_action = self.agent.actor_local(imagined_state.contiguous().view(b*t,-1)).to(state.device)
            action_loss = F.mse_loss(imagined_action, agent_action)
            
            # 3. Goal consistency loss
            stacked_imagined_state = torch.concatenate([imagined_state[:,1:,:], imagined_state[:,:-1,:]], dim=-1)
            trajectory_encoding = self.approx_embed_model(stacked_imagined_state)
            goal_embedding = self.goals[i]
            # imagine_state = trajectory_encoding[:,i,...].reshape(-1, trajectory_encoding.shape[-1])
            cos_sim = F.cosine_similarity(trajectory_encoding, goal_embedding.unsqueeze(0).unsqueeze(0))
            class_loss += 1-torch.mean(cos_sim)
            # Aggregate losses
            total_loss += 1.00 * class_loss + 0.50 * prox_loss + 0.50 * action_loss
            class_loss += class_loss
            policy_loss += action_loss
            proximity_loss += prox_loss

        # Return average loss across all imagined states
        return total_loss, class_loss, policy_loss, proximity_loss 
        
    def forward(self, next_state, caption):
        """
        Forward pass through the network. Given a state, outputs N imagined states.
        
        Args:
            state (Tensor): Input state of shape [batch_size, state_dim].
        Returns:
            Tensor: Imagined states of shape [batch_size, N, state_dim].
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(next_state).float().unsqueeze(0).to(device)
        features = self.encoder(state)
        film_layer_out = self.film_layer(features, caption)
        differential_state1 = self.fc1(film_layer_out)
        differential_state2 = self.fc2(film_layer_out)
        return torch.stack([differential_state1 + state, differential_state2 + state], dim = 1)
    
    def save(self, path):
        """
        Save the model's state dictionary.

        Args:
            path (str): File path to save the model.
        """
        params = {
        "feature_net": self.encoder.state_dict(),
        "fc1" : self.fc1.state_dict(),
        "fc2" : self.fc2.state_dict(),
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
        self.encoder.load_state_dict(params["feature_net"])
        self.fc1.load_state_dict(params["fc1"])
        self.fc2.load_state_dict(params["fc2"])
        print(f"[INFO] Model loaded from {path}")