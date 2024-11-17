import torch
import wandb
import torch.nn as nn
from utils.utils import *
import torch.optim as optim
from architectures.mlp import MLP
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from env.env import SimplePickup, MiniGridTransitionDescriber
from transformers import T5ForConditionalGeneration, T5Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = SimplePickup(max_steps=20, agent_view_size=5, size=7)
transition_captioner = MiniGridTransitionDescriber(5)

num_data = 10000

# Dataset Class
class FeatureToTextDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length=20):
        """
        Args:
            features (torch.Tensor): Feature tensors of shape (num_samples, feature_dim).
            labels (list): Corresponding natural language descriptions.
            tokenizer: Tokenizer for the text model.
            max_length (int): Maximum sequence length for tokenized labels.
        """
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # Tokenize the label
        tokenized_label = self.tokenizer(
            label,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "feature": feature,
            "input_ids": tokenized_label.input_ids.squeeze(0),
            "attention_mask": tokenized_label.attention_mask.squeeze(0),
            "language" : label
        }


# Model Class
class FeatureToTextModel(nn.Module):
    def __init__(self, feature_dim, max_timesteps=20, input_size=768, text_model_name="t5-small"):
        super(FeatureToTextModel, self).__init__()
        self.max_timesteps = max_timesteps
        self.input_size = input_size

        # Linear layer to map features to sequence format
        self.feature_encoder = nn.Sequential(
                                            nn.Linear(feature_dim, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, max_timesteps * input_size)
                                            )
        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)

        # Add padding token if not present
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.text_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, features, labels=None, attention_mask=None):
        # Encode feature tensor
        encoded_features = self.feature_encoder(features)  # Shape: (batch_size, max_timesteps * input_size)
        encoded_features = encoded_features.view(features.size(0), self.max_timesteps, self.input_size)
        encoded_features = (encoded_features - encoded_features.mean(dim=1, keepdim=True)) / (encoded_features.std(dim=1, keepdim=True) + 1e-8)

        # Pass through the T5 model
        if labels is not None:
            outputs = self.text_model(
                inputs_embeds=encoded_features,
                labels=labels,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.text_model(inputs_embeds=encoded_features)

        return outputs

    def generate(self, features, max_length=20):
        # Encode feature tensor
        with torch.no_grad():
            encoded_features = self.feature_encoder(features)
            encoded_features = encoded_features.view(features.size(0), self.max_timesteps, self.input_size)
            encoded_features = (encoded_features - encoded_features.mean(dim=1, keepdim=True)) / (encoded_features.std(dim=1, keepdim=True) + 1e-8)

        # Generate sequence
        generated_ids = self.text_model.generate(
            inputs_embeds=encoded_features,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            do_sample=True,  # Enable sampling
            top_k=20,        # Restrict to top 50 tokens
            top_p=0.1,      # Use nucleus sampling
            temperature=0.1, # Introduce randomness
            )

        # Decode generated text
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions



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
        caption_encoding = sentence_encoder.encode(transition_caption, convert_to_tensor=True, device=device)
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
    max_timesteps = 20
    input_size = 512
    features, labels, captions = collect_data(max_timestep=num_data)
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Dataset and DataLoader
    dataset = FeatureToTextDataset(features, labels, tokenizer, max_length=max_timesteps)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # Initialize Model
    model = FeatureToTextModel(feature_dim=feature_dim, max_timesteps=max_timesteps, input_size=input_size)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    # for param in model.text_model.parameters():
    #     param.requires_grad = False
    # Training Loop
    for epoch in range(500):  # Number of epochs
        model.train()
        total_loss = 0

        for batch in dataloader:
            features = batch["feature"].to(device)
            labels = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(features.float().to(device), labels, attention_mask)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        wandb.log({"loss" : np.mean(total_loss)}, step=epoch)
    torch.save(model.state_dict(), 'models/captioner.pth')
    
if __name__ == "__main__":
    sentence_encoder = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    main()