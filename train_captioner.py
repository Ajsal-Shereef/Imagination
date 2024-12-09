import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import torch.optim as optim
# from architectures.mlp import MLP
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from env.env import SimplePickup, MiniGridTransitionDescriber
from transformers import T5ForConditionalGeneration, T5Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = SimplePickup(max_steps=20, agent_view_size=5, size=7)
transition_captioner = MiniGridTransitionDescriber(5)
sentence_encoder = SentenceTransformer("all-MiniLM-L12-v2", device=device)

num_data = 10000


# Dataset Class
class FeatureToTextDataset(Dataset):
    def __init__(self, features, captions_encoded):
        """
        Args:
            features (torch.Tensor): Feature tensors of shape (num_samples, feature_dim).
            labels (list): Corresponding natural language descriptions.
            tokenizer: Tokenizer for the text model.
            max_length (int): Maximum sequence length for tokenized labels.
        """
        self.features = features
        self.labels = captions_encoded
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label


# Model Class
class FeatureToEmbeddingModel(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(FeatureToEmbeddingModel, self).__init__()
        self.input_size = feature_dim
        self.latent_dim = latent_dim

        # Linear layer to map features to sequence format
        self.feature_encoder = nn.Sequential(
                                            nn.Linear(self.input_size, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.latent_dim)
                                            )

    def forward(self, features):
        # Encode feature tensor
        encoded_features = self.feature_encoder(features)  # Shape: (batch_size, max_timesteps * input_size)
        return encoded_features
    

def collect_data(max_timestep):
    input = []
    target = []
    captions = []
    step = 0
    state, info = env.reset()
    while step<=max_timestep:
        action = env.action_space.sample()
        p_agent_loc = env.agent_pos
        p_state = env.get_unprocesed_obs()
        next_state, reward, terminated, truncated, _ = env.step(action)
        c_state = env.get_unprocesed_obs()
        c_agent_loc = env.agent_pos
        data = np.concatenate([state, next_state], axis = -1)
        transition_caption = transition_captioner.generate_description(agent_prev_pos = p_agent_loc, 
                                                                       agent_curr_pos = c_agent_loc, 
                                                                       agent_prev_dir = p_state['direction'], 
                                                                       agent_curr_dir = c_state['direction'], 
                                                                       prev_view = p_state['image'],
                                                                       curr_view = c_state['image'],
                                                                       purple_key_pos = (2,4), 
                                                                       green_ball_pos = (4,2),
                                                                       agent_action = action)
        caption_encoding = sentence_encoder.encode(transition_caption, convert_to_tensor=True, device=device)
        captions.append(caption_encoding)
        done = terminated + truncated
        state = next_state
        input.append(data)
        target.append(transition_caption)
        step += 1
        if done:
            state, info = env.reset()
            
    # np.save("data/input.npy", input)
    # np.save("data/caption.npy", target)
    return input, target, captions

def main():
    wandb.init(project="Captioner Training")
    feature_dim = 158
    latent_dim = sentence_encoder.get_sentence_embedding_dimension()
    features, captions, captions_encoded  = collect_data(max_timestep=num_data)
    
    # Dataset and DataLoader
    dataset = FeatureToTextDataset(features, captions_encoded)
    dataloader = DataLoader(dataset, batch_size=300, shuffle=True)

    # Initialize Model
    model = FeatureToEmbeddingModel(feature_dim=feature_dim, latent_dim=latent_dim)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # for param in model.text_model.parameters():
    #     param.requires_grad = False
    # Training Loop
    for epoch in range(1000):  # Number of epochs
        model.train()
        total_loss = 0

        for features, labels  in dataloader:
            optimizer.zero_grad()
            outputs = model(features.float().to(device))
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        wandb.log({"loss" : np.mean(total_loss)}, step=epoch)
    torch.save(model.state_dict(), 'models/captioner.pth')
    
if __name__ == "__main__":
    main()