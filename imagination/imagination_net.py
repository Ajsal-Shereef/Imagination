import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from architectures.mlp import MLP, Linear
from architectures.Layers import *
from architectures.film import FiLMLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

class ImaginationNet(nn.Module):
    def __init__(self,
                 env,
                 config,
                 num_goals,
                 agent
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
        hidden_layers = config.hidden_layers
        self.encoder = MLP(input_size = self.input_dim,
                           output_size = config.feature_dim,
                           hidden_sizes = hidden_layers,
                           output_activation = F.relu,
                           dropout_prob = 0.0)
        self.fc1 = Linear(in_dim = config.feature_dim, 
                          out_dim = self.input_dim)
        self.fc2 = Linear(in_dim = config.feature_dim, 
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

        # Forward pass through VAE and SAC agent for original state
        # with torch.no_grad():
        _, state_inference_out = self.vae(state)  # Get VAE class probabilities
        agent_action = self.agent.get_action(state).to(state.device)  # Get agent's action for original state
        agent_action_one_hot = F.one_hot(agent_action, num_classes=self.env.action_space.n).float()

        # # Class probabilities from the VAE for the original state
        # state_class_prob = torch.exp(state_inference_out['log_c'])

        # # Prepare targets for the two classes (target distributions)
        # max_indices = state_class_prob.argmax(dim=1)
        # target1 = torch.zeros_like(state_class_prob)
        # target2 = torch.zeros_like(state_class_prob)

        # # Assign probabilities based on max indices
        # target1[max_indices == 0] = state_class_prob[max_indices == 0]
        # target2[max_indices == 0] = 1 - state_class_prob[max_indices == 0]
        # target2[max_indices == 1] = state_class_prob[max_indices == 1]
        # target1[max_indices == 1] = 1 - state_class_prob[max_indices == 1]

        # # Stack targets for each imagined state
        # target_distributions = torch.stack((target1, target2), dim=0)

        # Loss for each imagined state
        for i in range(self.num_goals):
            imagined_state = imagined_states[:, i, :]

            # 1. Negative log-likelihood loss for class consistency
            # with torch.no_grad():
            _, imagined_inference_out = self.vae(imagined_state)
            imagined_class_prob = torch.exp(imagined_inference_out['log_c'])
            target = torch.zeros_like(imagined_class_prob)
            target[:,i] = 1
            # cl_loss = F.cross_entropy(imagined_class_prob.log(), target_distributions[i])
            cl_loss = F.cross_entropy(imagined_class_prob.log(), target)

            # 2. Proximity loss between imagined state and original state
            prox_loss = F.mse_loss(imagined_state, state)

            # 3. Action consistency loss with SAC policy
            # with torch.no_grad():
            imagined_action = self.agent.get_action(imagined_state).to(state.device)
            imagined_action_one_hot = F.one_hot(imagined_action, num_classes=self.env.action_space.n).float()
            action_loss = F.mse_loss(imagined_action_one_hot, agent_action_one_hot)

            # Aggregate losses
            total_loss += 1.00*cl_loss + 0.50 * prox_loss + 0.50 * action_loss
            class_loss += cl_loss
            policy_loss += action_loss
            proximity_loss += prox_loss

        # Return average loss across all imagined states
        return total_loss, class_loss, policy_loss, proximity_loss

    
    # # Loss function: 
    # def compute_loss(self, state, imagined_states):
    #     """
    #     Computes the total loss for the imagined states. The loss consists of:
    #     1. Cross-entropy between VAE output and one-hot vector for Gaussian component.
    #     2. L2 norm between imagined states and input state.
    #     3. Cross-entropy between SAC policy for imagined states and the action taken.

    #     Args:
    #         state (Tensor): Input state of shape [batch_size, state_dim].
    #         m (float): Coefficient for the policy loss term.
        
    #     Returns:
    #         Tensor: Computed loss.
    #     """
    #     total_loss = 0.0
    #     class_loss = 0.0
    #     policy_loss = 0.0
    #     proximity_loss = 0.0
    #     with torch.no_grad():
    #         _, state_inference_out = self.vae(state) #Forward pass through VAE
    #         agent_action = self.agent.get_action(state).to(device) #Forward pass through Agent
    #     agent_action = torch.nn.functional.one_hot(agent_action, self.env.action_space.n).float()
    #     state_class_prob = torch.exp(state_inference_out['log_c'])

    #     # Identify the index of the highest probability in each row
    #     max_indices = state_class_prob.argmax(dim=1)  # 0 if A[:,0] is higher, 1 if A[:,1] is higher
        
    #     # Initialize B and C with the same shape as A
    #     target1 = torch.zeros_like(state_class_prob)
    #     target2 = torch.zeros_like(state_class_prob)
        
    #     # Construct B by copying values where max is at the original positions in A
    #     target1[max_indices == 0] = state_class_prob[max_indices == 0]
    #     target2[max_indices == 0] = 1-state_class_prob[max_indices == 0]

    #     # Construct C by inverting the positions of max and min values of A
    #     target2[max_indices == 1] = state_class_prob[max_indices == 1]
    #     target1[max_indices == 1] = 1-state_class_prob[max_indices == 1]
        
    #     target = torch.stack((target1,target2))

    #     # Loop over each imagined state
    #     for i in range(self.num_goals):
    #         imagined_state = imagined_states[:, i, :] + state
    #         # 1. Cross-entropy between VAE class prob and i'th class
    #         with torch.no_grad():
    #             _, inference_out = self.vae(imagined_state) #Forward pass through VAE
    #         class_prob = torch.exp(inference_out['log_c']) #Class probabilities from VAE
    #         cl_loss = F.cross_entropy(class_prob, target[i])

    #         # 2. L2 distance between imagined state and input state
    #         prox_loss = F.mse_loss(imagined_state, state)

    #         # 3. Cross-entropy between SAC policy and one-hot vector of action taken
    #         with torch.no_grad():
    #             imagined_state_action = self.agent.get_action(imagined_state).to(device)
    #         imagined_state_action = torch.nn.functional.one_hot(imagined_state_action, self.env.action_space.n).float()
            
    #         action_loss = F.mse_loss(imagined_state_action, agent_action)

    #         # Total loss for this imagined state
    #         total_loss += cl_loss + 0.10*prox_loss + 0.10*action_loss 
            
    #         class_loss += cl_loss
    #         policy_loss += action_loss
    #         proximity_loss += prox_loss
            
    #     # Return the average loss across all N imagined states
    #     return total_loss, class_loss, policy_loss, proximity_loss 
        
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
        features = self.encoder(state)
        differential_state1 = self.fc1(features)
        differential_state2 = self.fc2(features)
        return torch.stack([differential_state1 + state, differential_state2 + state], dim = 1) #[batch_size, 2, delta_dim]
    
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