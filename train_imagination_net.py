import yaml
import torch
import wandb
import argparse
import gymnasium
import numpy as np
from tqdm import tqdm
from utils.utils import *
from vae.vae import GMVAE
import torch.optim as optim
from sac_agent.agent import SAC
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer
from imagination.imagination_net import ImaginationNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--env", type=str, default="SimplePickup",
                        help="Environement to use")
    parser.add_argument("--max_ep_len", type=int, default=20, help="Number of timestep within an episode")
    parser.add_argument("--encoder_model", default="all-MiniLM-L6-v2",
                        help="which model to use as encoder")
    parser.add_argument("--datapath", default="data/data.pkl",
                        help="Dataset to train the imagination net")
    parser.add_argument("--imagination_net_config", default="config/imagination_net.yaml",
                        help="Imagination net config directory")
    parser.add_argument("--vae_checkpoint", default="models/all-MiniLM-L6-v2/Feature_based/vae_epoch_1000.pth",
                        help="VAE checkpoint path")
    parser.add_argument("--sac_agent_checkpoint", default="models/sac_agent/SAC_discrete.tar",
                        help="SAC agent checkpoint path")
    parser.add_argument("--buffer_size", type=int, default=3000_00, help="Maximal training dataset size, default: 100_000")
    return parser.parse_args()

def train_imagination_net(config, 
                          vae, 
                          sac_agent, 
                          dataloader, 
                          checkpoint_interval, 
                          checkpoint_dir,
                          input_dim,
                          num_goals):
    hidden_layers = config['hidden_layers']
    imagination_net = ImaginationNet(input_dim = input_dim,
                                     hidden_layers = hidden_layers,
                                     num_goals = num_goals,
                                     vae = vae,
                                     sac = sac_agent).to(device)
    wandb.watch(imagination_net)
    #Creating the optimizer
    optimizer = optim.Adam(imagination_net.parameters(), lr=config['lr'])
    # nn_utils.clip_grad_norm_(imagination_net.parameters(), max_norm=config['max_gradient'])
    for epoch in range(config['epoch']):
        total_loss = 0
        total_class_loss = 0
        total_policy_loss = 0
        total_proximity_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['epoch']}")
        for batch in pbar:
            data = batch.float().to(device)
            optimizer.zero_grad()
            loss, class_loss, policy_loss, proximity_loss  = imagination_net.compute_loss(data)
            loss.backward()
            optimizer.step()
            # Accumulate losses
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_policy_loss += policy_loss.item()
            total_proximity_loss += proximity_loss.item()
            pbar.set_postfix({'Loss': loss.item(), 'Class loss': class_loss.item(), 'Policy loss': policy_loss.item(), 'Proximity loss': proximity_loss.item()})
            
        average_loss = total_loss / len(dataloader)
        average_class_loss = total_class_loss / len(dataloader)
        average_policy_loss = total_policy_loss / len(dataloader)
        average_proximity_loss = total_proximity_loss / len(dataloader)
        # current_lr = scheduler.get_last_lr()[0]
        wandb.log({"Total loss" : average_loss,
                   "Class loss" : average_class_loss,
                   "Policy loss" : average_policy_loss,
                   "Proximity loss" : average_proximity_loss}, step = epoch)
        # Save checkpoint at specified intervals
        if epoch % checkpoint_interval == 0 or epoch == config['epoch']:
            checkpoint_path = os.path.join(checkpoint_dir, f'imagination_net_epoch_{epoch}.pth')
            imagination_net.save(checkpoint_path)
        # print(f"Epoch {epoch}/{epochs} - Loss: {average_loss:.4f}, Recon Loss: {average_recon:.4f}, KL Divergence: {average_kl:.4f}, KL Weight: {kl_weight:.4f}")
    print("Training complete.")

def main(args):
    dataset = get_data(args.datapath)
    if args.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.max_ep_len, agent_view_size=5, size=7)
    # dataset = normalize_data(dataset)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the config file for imagination_net
    with open('config/imagination_net.yaml', 'r') as file:
        imagination_config = yaml.safe_load(file)

    # Initialize dataset and dataloader
    dataset = TextDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=imagination_config['batch_size'], shuffle=True)
        
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
    #Loading the pretrained weight of sac agent.
    sac_agent.load_params(args.sac_agent_checkpoint)
    #Freezing the SAC agent weight to prevent updating
    for params in sac_agent.parameters():
        params.requires_grad = False
    
    # Create data directory if it doesn't exist
    model_dir = 'models/imagination_net'
    os.makedirs(model_dir, exist_ok=True)
    
    #Train imagination Net
    train_imagination_net(config = imagination_config, 
                          vae = vae, 
                          sac_agent = sac_agent, 
                          dataloader = dataloader, 
                          checkpoint_interval = 200, 
                          checkpoint_dir = model_dir,
                          input_dim = env.observation_space.shape[0],
                          num_goals = len(goals))
    
if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="Imagination-net_training", config=args)
    main(args)
    