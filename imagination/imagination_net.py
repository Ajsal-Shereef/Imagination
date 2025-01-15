import cv2
import json
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from architectures.Layers import *
from architectures.film import FiLMLayer
from architectures.mlp import MLP, Linear
from architectures.cnn import CNNLayer, CNN
from env.env import MiniGridTransitionDescriber
from sentence_transformers import SentenceTransformer
from helper_functions.utils import anneal_coefficient, compute_entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

class ImaginationNet(nn.Module):
    def __init__(self,
                 env,
                 config,
                 num_goals,
                 agent,
                 vae
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
        self.input_dim = env.observation_space['image'].shape[-1]
        self.vae = vae
        hidden_layers = config.Imagination_Network.hidden_layers
        conv1 = CNNLayer(self.input_dim, 32, 5, 2)
        conv2 = CNNLayer(32, 64, 5, 2)
        conv3 = CNNLayer(64, 64, 3)
        conv_feature = Linear(1600, config.Imagination_Network.output_dim)
        self.feature_encoder = CNN([conv1, conv2, conv3], conv_feature)
        
        self.fc_decoder1 = Linear(config.Imagination_Network.output_dim, 256)
        self.fc_decoder2 = Linear(256, 1600)
        
        self.deconv1 = nn.Sequential(
            # First layer: Upsample to 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),  # 64x3x3 -> 32x16x16
            nn.LeakyReLU(),
            
            # Second layer: Upsample to 40x40
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),  # 32x16x16 -> 3x40x40
            nn.LeakyReLU(),  # Optional: Apply activation for output normalization
            
            # Third layer: Upsample to 40x40
            nn.ConvTranspose2d(16, self.input_dim, kernel_size=4, stride=2, padding=0),  # 32x16x16 -> 3x40x40
            nn.LeakyReLU()  # Optional: Apply activation for output normalization
        )
        
        self.deconv2 = nn.Sequential(
            # First layer: Upsample to 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),  # 64x3x3 -> 32x16x16
            nn.LeakyReLU(),
            
            # Second layer: Upsample to 40x40
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),  # 32x16x16 -> 3x40x40
            nn.LeakyReLU(),  # Optional: Apply activation for output normalization
            
            # Third layer: Upsample to 40x40
            nn.ConvTranspose2d(16, self.input_dim, kernel_size=4, stride=2, padding=0),  # 32x16x16 -> 3x40x40
            nn.LeakyReLU()  # Optional: Apply activation for output normalization
        )
        
        self.agent = agent
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.num_goals = num_goals
        self.centroid = self.config.Imagination_General.centroid_Path
        #Open the JSON file and load its content 
        with open(self.centroid, 'r') as file: 
            self.centroid = json.load(file)
        #Preventing agent to take random action.
        # if config.General.agent == 'dqn':
        #     self.agent.set_episode_step()

    def compute_loss(self, state, imagined_states, epoch):
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
        centroid_losses = 0.0
        # Forward pass through SAC agent for original state
        agent_action = self.agent.actor_local(state).to(state.device)  # Get agent's action for original state
        # Class probabilities from the VAE for the original state
        _, _, _, _, state_class_prob = self.vae(state)

        # Prepare targets for the two classes (target distributions)
        max_indices = state_class_prob.argmax(dim=1)
        target1 = torch.zeros_like(state_class_prob)
        target2 = torch.zeros_like(state_class_prob)

        # Assign probabilities based on max indices
        target1[max_indices == 0] = state_class_prob[max_indices == 0]
        target2[max_indices == 0] = 1 - state_class_prob[max_indices == 0]
        target2[max_indices == 1] = state_class_prob[max_indices == 1]
        target1[max_indices == 1] = 1 - state_class_prob[max_indices == 1]

        # # Stack targets for each imagined state
        target_distributions = torch.stack((target1, target2), dim=0)
        
        # entropy = compute_entropy(state_class_prob.detach().cpu().numpy())
        # max_entropy = np.log(state_class_prob.shape[-1])
        # Weighting factor (inverse of normalized entropy)
        # weight = 1 - (entropy / max_entropy)
        # weight = torch.tensor(weight).unsqueeze(-1).to(device)
        # Loss for each imagined state
        for i in range(self.num_goals):
            imagined_state = imagined_states[:, i, :]
            # 1. Proximity loss between imagined state and original state
            prox_loss = F.mse_loss(imagined_state, state)

            # 2. Action consistency loss with SAC policy
            imagined_action = self.agent.actor_local(imagined_state).to(state.device)
            action_loss = F.mse_loss(imagined_action, agent_action)
            
            # 3. Class consistency loss
            # target = torch.stack([F.one_hot(torch.tensor(i), num_classes=self.num_goals).float() for _ in range(imagined_state.shape[0])]).to(device)
            # target = weight * target + (1 - weight) * torch.full_like(target, 1/self.num_goals).to(device)
            _, img_z, _, _, imagined_state_class = self.vae(imagined_state)
            # class_consistency_loss = F.binary_cross_entropy(imagined_state_class, target)
            class_consistency_loss = F.binary_cross_entropy(imagined_state_class, target_distributions[i,...])
            
            # #Forcing the centroid of the imagined state to the centroid output from GMM
            # centroid = self.centroid[f'Goal_{i}']
            # centroid_loss = F.mse_loss(torch.tensor(centroid).unsqueeze(0).to(device), img_z)
            
            # Aggregate losses
            # current_weight = anneal_coefficient(epoch, self.config.Imagination_Network.epoch, 0.5, 1, 100)
            current_weight = 0.0
            # total_loss += 1.00 * class_consistency_loss + current_weight * prox_loss + current_weight * action_loss
            total_loss += 1.00 * class_consistency_loss + 0.10 * prox_loss + current_weight * action_loss
            class_loss += class_consistency_loss
            policy_loss += action_loss
            proximity_loss += prox_loss
            # centroid_losses += centroid_loss
        # Return average loss across all imagined states
        return total_loss, class_loss, policy_loss, proximity_loss, current_weight
        
    def forward(self, state):
        """
        Forward pass through the network. Given a state, outputs N imagined states.
        
        Args:
            state (Tensor): Input state of shape [batch_size, state_dim].
        Returns:
            Tensor: Imagined states of shape [batch_size, N, state_dim].
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float().unsqueeze(0).to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        encoder_features = self.feature_encoder(state)
        decoder_input = self.fc_decoder1(encoder_features[0])
        decoder_input = self.fc_decoder2(decoder_input)
        decoder_input = decoder_input.view(decoder_input.shape[0], 64, 5, 5)
        differential_state1 = self.deconv1(decoder_input)
        differential_state2 = self.deconv2(decoder_input)
        return torch.stack([differential_state1, differential_state2], dim = 1)
    
    def save(self, path):
        """
        Save the model's state dictionary.

        Args:
            path (str): File path to save the model.
        """
        params = {
        "feature_encoder": self.feature_encoder.state_dict(),
        "fc_decoder1" : self.fc_decoder1.state_dict(),
        "fc_decoder2" : self.fc_decoder2.state_dict(),
        "deconv1" : self.deconv1.state_dict(),
        "deconv2" : self.deconv2.state_dict(),
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
        self.feature_encoder.load_state_dict(params["feature_encoder"])
        self.fc_decoder1.load_state_dict(params["fc_decoder1"])
        self.fc_decoder2.load_state_dict(params["fc_decoder2"])
        self.deconv1.load_state_dict(params["deconv1"])
        self.deconv2.load_state_dict(params["deconv2"])
        print(f"[INFO] Model loaded from {path}")