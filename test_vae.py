import os
import hydra
import torch
import numpy as np
from vae.vae import GMVAE
from utils.utils import *
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer

@hydra.main(version_base=None, config_path="config", config_name="vae")
def test_vae(args: DictConfig):
    vae_checkpoint, datapath = "models/gmvae/vae_epoch_6800.pth", f"data/{args.General.env}/data.pkl"
    dataset = get_data(datapath)
    # dataset = normalize_data(dataset)
    # Initialize dataset and dataloader
    dataset = TextDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    # goal_gen = GetLLMGoals()
    # goals = goal_gen.llmGoals([])
    # # Load the sentecebert model to get the embedding of the goals from the LLM
    # sentencebert = SentenceTransformer(pretrained_model)
    # # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    # mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    # latent_dim = sentencebert.get_sentence_embedding_dimension()
    num_mixtures = args.General.num_goals
    vae = GMVAE(
                input_dim = dataset.sentences[0].shape[0], 
                encoder_hidden = args.Network.encoder_hidden,
                encoder_out = args.Network.encoder_out,
                decoder_hidden = args.Network.decoder_hidden, 
                latent_dim=args.Network.latent_dim, 
                num_mixtures=num_mixtures
                )
    vae.load(vae_checkpoint)
    vae.to(device)
    vae.eval()
    # Visualization of the latent space 
    data_dir = f'visualizations/GMVAE/'
    os.makedirs(data_dir, exist_ok=True)
    latent, labels = visualize_latent_space(vae, dataloader, device, method='pca', save_path=data_dir, checkpoint = vae_checkpoint.split("/")[-1])
    latent = visualize_latent_space(vae, dataloader, device, latent, labels, method='tsne', save_path=data_dir, checkpoint = vae_checkpoint.split("/")[-1])
    
if __name__ == "__main__":
    test_vae()
    
    