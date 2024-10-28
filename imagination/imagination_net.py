import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from architectures.mlp import MLP
from architectures.Layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

class ImaginationNet(nn.Module):
    def __init__(self,
                 env,
                 hidden_layers,
                 num_goals,
                 vae,
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
        self.imagination_net = MLP(input_size = self.input_dim,
                                   output_size = num_goals*self.input_dim,
                                   hidden_sizes = hidden_layers,
                                   output_activation = F.relu)
        self.vae = vae
        self.vae.eval()
        self.agent = agent
        self.agent.eval()
        self.num_goals = num_goals
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    # Loss function: 
    def compute_loss(self, state, m=0.5):
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
        imaginated_state = self(state)
        # q1_original = self.sac_agent.critic1(state)
        # q2_original = self.sac_agent.critic2(state)
        # q_values_original = torch.min(q1_original,q2_original)
        total_loss = 0.0
        class_loss = 0.0
        policy_loss = 0.0
        proximity_loss = 0.0
        with torch.no_grad():
            inference_state_out, _ = self.vae(state) #Forward pass through VAE
            action = self.agent.get_action(state) #Forward pass through Agent
        state_class_prob = inference_state_out['prob_cat']

        # Identify the index of the highest probability in each row
        max_indices = state_class_prob.argmax(dim=1)  # 0 if A[:,0] is higher, 1 if A[:,1] is higher

        
        # Initialize B and C with the same shape as A
        target1 = torch.zeros_like(state_class_prob)
        target2 = torch.zeros_like(state_class_prob)
        
        # Construct B by copying values where max is at the original positions in A
        target1[max_indices == 0] = state_class_prob[max_indices == 0]
        target2[max_indices == 0] = 1-state_class_prob[max_indices == 0]

        # Construct C by inverting the positions of max and min values of A
        target2[max_indices == 1] = state_class_prob[max_indices == 1]
        target1[max_indices == 1] = 1-state_class_prob[max_indices == 1]
        
        target = torch.stack((target1,target2))

        # Loop over each imagined state
        for i in range(self.num_goals):
            imagined_state = imaginated_state[:, i, :]
            # 1. Cross-entropy between VAE output and one-hot vector for Gaussian component i
            with torch.no_grad():
                inference_imagined_state_out, _ = self.vae(imagined_state) #Forward pass through VAE
            imagined_state_class_prob = inference_imagined_state_out["prob_cat"] #Class probabilities from VAE
            ce_loss_vae = self.mse_loss(imagined_state_class_prob, target[i])

            # 2. L2 distance between imagined state and input state
            l2_loss = self.mse_loss(imagined_state, state)

            # 3. Cross-entropy between SAC policy and one-hot vector of action taken
            with torch.no_grad():
                imagined_state_action = self.agent.get_action(imagined_state)
                
            # Create a one-hot tensor
            one_hot_action = torch.zeros(self.env.action_space.n)
            one_hot_action[action] = 1
            one_hot_imagined_action = torch.zeros(self.env.action_space.n)
            one_hot_imagined_action[imagined_state_action] = 1
                
                # q1 = self.sac_agent.critic1(imagined_state)
                # q2 = self.sac_agent.critic2(imagined_state)
                # q_values = torch.min(q1,q2)
            # one_hot_action = torch.zeros_like(policy_imagined)
            # one_hot_action.scatter_(1, sac_action.unsqueeze(1), 1)  # One-hot for action taken

            ce_loss_policy = self.mse_loss(one_hot_action, one_hot_imagined_action)

            # Total loss for this imagined state
            total_loss += ce_loss_vae + 0.5*l2_loss + ce_loss_policy * m
            
            class_loss += ce_loss_vae
            policy_loss += ce_loss_policy
            proximity_loss += l2_loss
            
        # Return the average loss across all N imagined states
        return total_loss, class_loss, policy_loss, proximity_loss 
        
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
        imagined_state = self.imagination_net(state).view(-1, self.num_goals, self.input_dim)
        return imagined_state
    
    def save(self, path):
        """
        Save the model's state dictionary.

        Args:
            path (str): File path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"[INFO] Model saved to {path}")

    def load(self, path):
        """
        Load the model's state dictionary.

        Args:
            path (str): File path from which to load the model.
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"[INFO] Model loaded from {path}")