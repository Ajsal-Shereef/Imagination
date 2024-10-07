import os
import torch
import numpy as np
from vae.vae import VAE
from utils.utils import *
from torch.utils.data import DataLoader
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer

def test_vae(vae_checkpoint, pretrained_model, datapath):
    dataset = get_data(datapath)
    # dataset = normalize_data(dataset)
    # Initialize dataset and dataloader
    dataset = TextDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    goal_gen = GetLLMGoals()
    goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    sentencebert = SentenceTransformer(pretrained_model)
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    learn_mu_p = False  # Enable learnable prior means
    latent_dim = sentencebert.get_sentence_embedding_dimension()
    vae = VAE(
        pretrained_model_name="all-MiniLM-L6-v2",
        latent_dim=latent_dim,
        num_mixtures=2,
        mu_p=mu_p,
        learn_mu_p=learn_mu_p
    )
    vae.load(vae_checkpoint)
    vae.to(device)
    # Visualization of the latent space 
    data_dir = f'visualizations/{pretrained_model}'
    os.makedirs(data_dir, exist_ok=True)
    latent = visualize_latent_space(vae, dataloader, device, method='pca', save_path=f'{data_dir}/latent_space_pca_{vae_checkpoint.split("/")[-1]}.png')
    latent = visualize_latent_space(vae, dataloader, device, latent, method='tsne', save_path=f'{data_dir}/latent_space_tsne_{vae_checkpoint.split("/")[-1]}.png')
    
if __name__ == "__main__":
    test_vae("models/all-MiniLM-L6-v2/vae_epoch_600.pth", "all-MiniLM-L6-v2", "data/captions.pkl")
    
    