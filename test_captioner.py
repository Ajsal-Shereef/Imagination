import torch
from utils.utils import *
from torch.utils.data import DataLoader
from train_captioner import FeatureToTextModel, FeatureToTextDataset, collect_data
from env.env import SimplePickup, MiniGridTransitionDescriber
from transformers import T5ForConditionalGeneration, T5Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_data = 50
feature_dim = 158
input_size = 512
max_timesteps = 20

model = FeatureToTextModel(feature_dim=feature_dim, max_timesteps=max_timesteps, input_size=input_size)
model.to(device)
model.load_state_dict(torch.load("models/captioner.pth", map_location=device))
features, labels = collect_data(num_data)
# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# Dataset and DataLoader
dataset = FeatureToTextDataset(features, labels, tokenizer, max_length=max_timesteps)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# Initialize Model

def test_model(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            features = batch["feature"].to(device)
            labels = batch["language"]
            caption = model.generate(features.float().unsqueeze(0))
            print("Generated Caption:", caption)
            print("Original caption:", labels)
            
test_model(model, dataloader)
