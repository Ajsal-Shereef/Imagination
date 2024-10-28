import yaml
import hydra
import torch
import wandb
import gymnasium
import numpy as np
from tqdm import tqdm
from utils.utils import *
from vae.vae import GMVAE
import torch.optim as optim
from dqn.dqn import DQNAgent
from sac_agent.agent import SAC
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer
from imagination.imagination_net import ImaginationNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_imagination_net(config,
                          env, 
                          vae, 
                          agent, 
                          dataloader, 
                          checkpoint_interval, 
                          checkpoint_dir,
                          num_goals):
    hidden_layers = config.hidden_layers
    imagination_net = ImaginationNet(env = env,
                                     hidden_layers = hidden_layers,
                                     num_goals = num_goals,
                                     vae = vae,
                                     agent = agent).to(device)
    # wandb.watch(imagination_net)
    #Creating the optimizer
    optimizer = optim.Adam(imagination_net.parameters(), lr=config['lr'])
    # nn_utils.clip_grad_norm_(imagination_net.parameters(), max_norm=config['max_gradient'])
    for epoch in range(config.epoch):
        total_loss = 0
        total_class_loss = 0
        total_policy_loss = 0
        total_proximity_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epoch}")
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
            
        average_loss = total_loss #/ len(dataloader)
        average_class_loss = total_class_loss #/ len(dataloader)
        average_policy_loss = total_policy_loss #/ len(dataloader)
        average_proximity_loss = total_proximity_loss #/ len(dataloader)
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

@hydra.main(version_base=None, config_path="config", config_name="imagination_net_master_config")
def main(args: DictConfig) -> None:
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    dataset = get_data(f'{args.General.datapath}/{args.General.encoder_model}/data.pkl')
    if args.General.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.General.max_ep_len, agent_view_size=5, size=7)
    # dataset = normalize_data(dataset)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    dataset = TextDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=args.Network.batch_size, shuffle=True)
        
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    goal_gen = GetLLMGoals()
    goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    sentencebert = SentenceTransformer(args.General.encoder_model)
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    # Define prior means (mu_p) for each mixture component
    # Initialize VAE with learnable prior means
    latent_dim = sentencebert.get_sentence_embedding_dimension()
    num_mixtures = len(goals)
    vae = GMVAE(
        input_dim = dataset[0].shape[0], 
        encoder_hidden = args.Network.encoder_hidden,
        decoder_hidden = args.Network.decoder_hidden, 
        latent_dim=latent_dim, 
        num_mixtures=num_mixtures, 
        mu_p=mu_p
    )
    #Loading the pretrained VAE
    vae.load(args.General.vae_checkpoint)
    vae.to(device)
    #Freezing the VAE weight to prevent updating
    for params in vae.parameters():
        params.requires_grad = False
        
    if args.General.agent == 'dqn':
        agent = DQNAgent(env, 
                        args.General, 
                        args.policy_config, 
                        args.policy_network_cfg, 
                        args.policy_network_cfg, '')
    else:
        agent = SAC(args,
                    state_size = env.observation_space.shape[0],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.General.buffer_size)
    
    #Loading the pretrained weight of sac agent.
    agent.load_params(args.General.agent_checkpoint)
    #Freezing the SAC agent weight to prevent updating
    for params in agent.parameters():
        params.requires_grad = False
    
    # Create data directory if it doesn't exist
    model_dir = 'models/imagination_net'
    os.makedirs(model_dir, exist_ok=True)
    
    #Train imagination Net
    train_imagination_net(config = args.Network, 
                          env= env,
                          vae = vae, 
                          agent = agent, 
                          dataloader = dataloader, 
                          checkpoint_interval = args.Network.checkpoint_interval, 
                          checkpoint_dir = model_dir,
                          num_goals = len(goals))
    
if __name__ == "__main__":
    wandb.init(project="Imagination-net_training")
    main()
    