import torch
import wandb
import argparse
import gymnasium
import numpy as np
from utils.utils import *
from vae.vae import GMVAE
from sac_agent.agent import SAC
from torch.utils.data import DataLoader
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer

def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--env", type=str, default="SimplePickup",
                        help="Environement to use")
    parser.add_argument("--use_logger", dest="use_logger", default=True,
                        help="whether store the results in logger")
    parser.add_argument("--encoder_model", default="all-MiniLM-L6-v2",
                        help="which model to use as encoder")
    parser.add_argument("--datapath", default="data/data.pkl",
                        help="Dataset to train the imagination net")
    parser.add_argument("--vae_checkpoint", default="models/all-MiniLM-L6-v2/Feature_based/vae_epoch_1000.pth",
                        help="VAE checkpoint path")
    parser.add_argument("--sac_agent_checkpoint", default="models/sac_agent/SAC_discrete.tar",
                        help="SAC agent checkpoint path")
    parser.add_argument("--epoch", default=1000,
                        help="Number of training iterations")
    parser.add_argument("--buffer_size", type=int, default=3000_00, help="Maximal training dataset size, default: 100_000")
    return parser.parse_args()

def train_iamgination_net(vae, sac_agent, dataloader, checkpoint_interval, checkpoint_dir):
    pass

def main(args):
    dataset = get_data(args.datapath)
    if args.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.max_ep_len, agent_view_size=5, size=7)
    # dataset = normalize_data(dataset)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    dataset = TextDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
        
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    goal_gen = GetLLMGoals()
    goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    sentencebert = SentenceTransformer(args.encoder_model)
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    # Define prior means (mu_p) for each mixture component
    # Initialize VAE with learnable prior means
    latent_dim = sentencebert.get_sentence_embedding_dimension()
    num_mixtures = len(goals)
    vae = GMVAE(
        input_dim = dataset[0].shape[0], 
        encoder_hidden = [1024,1024,512,512,512,256,256,256,256], #Don't forget to edit the snippet above as well
        decoder_hidden = [256,256,256,256,512,512,512,1024,1024], 
        latent_dim=latent_dim, 
        num_mixtures=num_mixtures, 
        mu_p=mu_p
    )
    #Loading the pretrained VAE
    vae.load(args.vae_checkpoint)
    vae.to(device)
    #Freezing the VAE weight to prevent updating
    for params in vae.parameters():
        params.requires_grad = False
    
    sac_agent = SAC(args,
                    state_size = env.observation_space.shape[0],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.buffer_size)
    #Loading the pretrained weight of sac agent. The SAC agent weights are already freezed while loading
    sac_agent.load_params(args.sac_agent_checkpoint)
    