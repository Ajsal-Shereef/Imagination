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
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # Prepend the start-of-text token
        label = f"<|startoftext|> {label}"
        
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
    def __init__(self, feature_dim, max_timesteps=20, input_size=768, text_model_name="gpt2"):
        super(FeatureToTextModel, self).__init__()
        self.max_timesteps = max_timesteps
        self.input_size = input_size

        # Linear layer to map features to sequence format
        self.feature_encoder = nn.Sequential(
                                            nn.Linear(feature_dim, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 8192),
                                            nn.ReLU(),
                                            nn.Linear(8192, max_timesteps * input_size)
                                            )
        self.text_model = GPT2LMHeadModel.from_pretrained(text_model_name)
        for params in self.text_model.parameters():
            params.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_model_name)

        # Add padding token if not present
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
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
    
    def generate(self, embed, entry_length=20, top_p=0.8, temperature=1.0):
        self.text_model.eval()
        captions = []
        generated_list = []
        # stop_token_index = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        pad_token_index = self.tokenizer.encode('[PAD]')[0]
        # filter_value = -float("Inf")
        with torch.no_grad():
            generated = embed
            encoded_features = self.feature_encoder(generated).view(-1, self.max_timesteps, self.input_size)
            encoded_features = (encoded_features - encoded_features.mean(dim=1, keepdim=True)) / (encoded_features.std(dim=1, keepdim=True) + 1e-8)
            batch_size, seq_len, _ = encoded_features.shape
            for batch_idx in range(batch_size):
                sequence_output = []  # Stores tokens for the current sequence
                # Extract the trajectory for the current batch
                trajectory = encoded_features[batch_idx]  # [seq_len, 768]
                trajectory = trajectory.unsqueeze(0)  # Add batch dimension -> [1, seq_len, 768]
                tokens = None
                for i in range(entry_length):
                    text_model_input = trajectory[:,:i+1,:]
                    outputs = self.text_model(inputs_embeds=text_model_input)
                    logits = outputs.logits
                    # logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    # sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    # cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # sorted_indices_to_remove = cumulative_probs > top_p
                    # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    # sorted_indices_to_remove[..., 0] = 0
                    # indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    # logits[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits[:,i,:], -1)
                    # next_token_embed = self.text_model.transformer.wte(next_token)
                    if next_token == pad_token_index:
                        break
                    if tokens is None: tokens = next_token
                    else: tokens = torch.cat((tokens, next_token), dim=-1)
                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = self.tokenizer.decode(output_list)
                generated_list.append(output_text)
        return generated_list

    # def generate(self, features, max_length=20):
    #     # Encode feature tensor
    #     with torch.no_grad():
    #         encoded_features = self.feature_encoder(features)
    #         encoded_features = encoded_features.view(features.size(0), self.max_timesteps, self.input_size)
    #         encoded_features = (encoded_features - encoded_features.mean(dim=1, keepdim=True)) / (encoded_features.std(dim=1, keepdim=True) + 1e-8)

    #     # Generate sequence
    #     generated_ids = self.text_model.generate(
    #         inputs_embeds=encoded_features,
    #         max_length=max_length,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         bos_token_id=self.tokenizer.bos_token_id,
    #         do_sample=True,  # Enable sampling
    #         top_k=20,        # Restrict to top 50 tokens
    #         top_p=0.1,      # Use nucleus sampling
    #         temperature=0.1, # Introduce randomness
    #         )

    #     # Decode generated text
    #     captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #     return captions



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
    max_timesteps = 20
    input_size = 768
    features, labels, captions = collect_data(max_timestep=num_data)
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Dataset and DataLoader
    dataset = FeatureToTextDataset(features, labels, tokenizer, max_length=max_timesteps)
    dataloader = DataLoader(dataset, batch_size=300, shuffle=True)

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
    for epoch in range(1000):  # Number of epochs
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
    main()