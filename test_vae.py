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
    num_mixtures = len(goals)
    vae = VAE(
        input_dim = dataset[0].shape[0], 
        encoder_hidden = [1024,1024,512,512,512,256,256,256],
        decoder_hidden = [256,256,256,512,512,512,1024,1024], 
        latent_dim=latent_dim, 
        num_mixtures=num_mixtures, 
        mu_p=mu_p, 
        learn_mu_p=learn_mu_p
    )
    vae.load(vae_checkpoint)
    vae.to(device)
    # Visualization of the latent space 
    data_dir = f'visualizations/{pretrained_model}/Feature_based'
    os.makedirs(data_dir, exist_ok=True)
    latent, labels = visualize_latent_space(vae, dataloader, device, method='pca', save_path=data_dir, checkpoint = vae_checkpoint.split("/")[-1])
    latent = visualize_latent_space(vae, dataloader, device, latent, labels, method='tsne', save_path=data_dir, checkpoint = vae_checkpoint.split("/")[-1])
    
if __name__ == "__main__":
    test_vae("models/all-MiniLM-L6-v2/Feature_based/vae_epoch_6000.pth", "all-MiniLM-L6-v2", "data/data.pkl")
    
    