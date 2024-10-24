import os
import hydra
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.utils import *
from vae.vae import GMVAE
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer


# =============================
# 1. Training Function
# =============================

def train_vae(model, dataloader, optimizer, device, checkpoint_dir, \
    epochs=10, kl_annealing=True, anneal_start=1, anneal_end=10, checkpoint_interval=200):
    """
    Train the VAE model.

    Args:
        model (VAE): The VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        epochs (int): Number of training epochs.
        kl_annealing (bool): If True, applies KL annealing.
        anneal_start (int): Epoch to start annealing.
        anneal_end (int): Epoch to end annealing.
    """
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_caption = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        # Determine KL weight
        if kl_annealing:
            if epoch < anneal_start:
                kl_weight = 1.0
            elif anneal_start <= epoch <= anneal_end:
                kl_weight = 1.0*((epoch - anneal_start + 1) / (anneal_end - anneal_start + 1))
            else:
                kl_weight = 1.0
        else:
            kl_weight = 1.0

        for data, caption in pbar:
            data = data.float().to(device)
            optimizer.zero_grad()
            # Forward pass
            inference_out, reconstruction = model(data)
            # Compute loss
            loss, recon_loss, kl = model.loss_function(data, caption, inference_out, reconstruction, kl_weight)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            # total_caption += caption_loss.item()
            pbar.set_postfix({'Loss': loss.item(), 'Recon': recon_loss.item(), 'KL': kl.item(), 'KL Weight': kl_weight})
        
        # Scheduler step
        # scheduler.step(total_loss)

        average_loss = total_loss / len(dataloader)
        average_recon = total_recon / len(dataloader)
        average_kl = total_kl / len(dataloader)
        # average_caption_loss = total_caption / len(dataloader)
        # current_lr = scheduler.get_last_lr()[0]
        wandb.log({"Total loss" : average_loss,
                   "Reconstruction loss" : average_recon,
                   "KL divergence" : average_kl,
                   "KL weight" : kl_weight}, step = epoch)
        # Save checkpoint at specified intervals
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'vae_epoch_{epoch}.pth')
            model.save(checkpoint_path)
        # print(f"Epoch {epoch}/{epochs} - Loss: {average_loss:.4f}, Recon Loss: {average_recon:.4f}, KL Divergence: {average_kl:.4f}, KL Weight: {kl_weight:.4f}")
    print("Training complete.")
    
# =============================
# 2. Training VAE
# =============================

@hydra.main(version_base=None, config_path="config", config_name="vae")
def main(args: DictConfig) -> None:
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    
    #Loading the dataset
    dataset = get_data(f'{args.General.datapath}/{args.General.encoder_model}/data.pkl')
    captions = get_data(f'{args.General.datapath}/{args.General.encoder_model}/captions.pkl')
    # dataset = normalize_data(dataset)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    data = TwoListDataset(dataset, captions)
    dataloader = DataLoader(data, batch_size=1000, shuffle=True)
        
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    goal_gen = GetLLMGoals()
    goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    sentencebert = SentenceTransformer(args.General.encoder_model)
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    # Define prior means (mu_p) for each mixture component
    latent_dim = sentencebert.get_sentence_embedding_dimension()
    num_mixtures = len(goals)

    vae = GMVAE(
        input_dim = dataset[0].shape[0], 
        encoder_hidden = args.Network.encoder_hidden, #Don't forget to edit the snippet above as well
        decoder_hidden = args.Network.decoder_hidden, 
        latent_dim=latent_dim, 
        num_mixtures=num_mixtures, 
        mu_p=mu_p
    )
    vae.to(device)

    # Define optimizer (only parameters that require gradients)
    optimizer = optim.Adam(vae.parameters(), lr=args.Network.lr)
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop with KL annealing
    epochs = args.Network.epoch
    kl_annealing = False
    anneal_start = args.Network.epoch/args.Network.epoch
    anneal_end = args.Network.epoch
    # Create data directory if it doesn't exist
    data_dir = f'models/{args.General.encoder_model}/Feature_based'
    os.makedirs(data_dir, exist_ok=True)
    train_vae(vae, dataloader, optimizer, device, data_dir, epochs, kl_annealing, anneal_start, anneal_end)

    # Save the trained model
    #save_path = f"{data_dir}/vae_sentence_bert_mog.pth"
    save_path = f"{data_dir}/vae_normal.pth"
    vae.save(save_path)

    # Visualization of the latent space 
    # data_dir = 'visualizations'
    # os.makedirs(data_dir, exist_ok=True)
    # latent = visualize_latent_space(vae, dataloader, device, method='pca', save_path=f'{data_dir}/latent_space_pca.png')
    # latent = visualize_latent_space(vae, dataloader, device, latent, method='tsne', save_path=f'{data_dir}/latent_space_tsne.png')

if __name__ == "__main__":
    wandb.init(project="Imagination-VAE_training")
    main()


