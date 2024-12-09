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
                 captioner,
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
        latent_dim = sentence_encoder.get_word_embedding_dimension()
        self.sentece_encoder = sentence_encoder
        self.captioner = captioner
        self.goals = goals
        hidden_layers = config.Imagination_Network.hidden_layers
        self.encoder = MLP(input_size = self.input_dim,
                           output_size = config.Imagination_Network.feature_dim,
                           hidden_sizes = hidden_layers,
                           output_activation = F.relu,
                           dropout_prob = 0.0)
        self.lstm = torch.nn.LSTM(input_size = config.Imagination_Network.feature_dim, hidden_size=config.Imagination_Network.lstm_hidden_size, batch_first=True)
        self.film_layer = FiLMLayer(config.Imagination_Network.feature_dim, latent_dim)
        self.fc1 = Linear(in_dim = config.Imagination_Network.lstm_hidden_size, 
                          out_dim = self.input_dim)
        self.fc2 = Linear(in_dim = config.Imagination_Network.lstm_hidden_size, 
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
            caption = self.captioner.generate(stacked_imagined_state)
            trajectory_encoding = self.sentece_encoder.encode(caption, convert_to_tensor=True, device=device).view(-1, 20, 384)
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
        
    def forward(self, state, caption, hx=None):
        """
        Forward pass through the network. Given a state, outputs N imagined states.
        
        Args:
            state (Tensor): Input state of shape [batch_size, state_dim].
        Returns:
            Tensor: Imagined states of shape [batch_size, N, state_dim].
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float().unsqueeze(0).to(device)
        b,t,f = state.shape
        input = state.view(b*t, -1)
        caption = caption.view(caption.shape[0]*caption.shape[1], -1)
        features = self.encoder(input)
        film_layer_out = self.film_layer(features, caption)
        film_layer_out = film_layer_out.view(b, t, -1)
        lstm_output, hx = self.lstm(film_layer_out, hx)
        lstm_output = lstm_output.contiguous().view(b*t, -1)
        differential_state1 = self.fc1(lstm_output)
        differential_state2 = self.fc2(lstm_output)
        return torch.stack([differential_state1.view(b,t,f) + state, differential_state2.view(b,t,f) + state], dim = 1), hx
    
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