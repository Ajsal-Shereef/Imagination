import torch
import wandb
import torch.nn as nn
from utils.utils import *
import torch.optim as optim
from architectures.mlp import MLP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from env.env import SimplePickup, MiniGridTransitionDescriber
from transformers import T5ForConditionalGeneration, T5Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = SimplePickup(max_steps=20, agent_view_size=5, size=7)
transition_captioner = MiniGridTransitionDescriber(5)

num_data = 20000

# Custom Dataset
class FeatureCaptionDataset(Dataset):
    def __init__(self, features, captions):
        self.features = features  # List of feature tensors
        self.captions = captions  # List of corresponding captions

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.captions[idx]

# CSurrogate Model
class SurrogateModel(nn.Module):
    def __init__(self, feature_dim, text_embedding_dim):
        super(SurrogateModel, self).__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, text_embedding_dim)  # Match text embedding dimension
        )

    def forward(self, features):
        # Encode features
        return self.feature_encoder(features)
    
    def load_params(self, path):
        """Load model and optimizer parameters."""
        params = torch.load(path, map_location=device)
        self.load_state_dict(params)
        print("[INFO] loaded the SAC model", path)

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
                                                                       red_ball_pos = (2,4), 
                                                                       green_ball_pos = (4,2),
                                                                       agent_action = action)
        caption_encoding = sentencebert.encode(transition_caption, convert_to_tensor=True, device=device)
        captions.append(caption_encoding)
        done = terminated + truncated
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
    features, labels, caption_encoding = collect_data(max_timestep=num_data)
    
    # Dataset and DataLoader
    dataset = FeatureCaptionDataset(features, caption_encoding)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    model = SurrogateModel(feature_dim=feature_dim, text_embedding_dim=384).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    for epoch in range(1000):  # Number of epochs
        total_loss = 0
        for features, captions in dataloader:
            features = features.to(device)
        
            # Forward pass
            optimizer.zero_grad()
            feature_embeddings = model(features.float())
            loss = criterion(feature_embeddings, captions)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        wandb.log({"loss" : np.mean(total_loss)}, step=epoch)
    torch.save(model.state_dict(), 'models/captioner.pth')
    
if __name__ == "__main__":
    sentencebert = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    main()