import hydra
import torch
import random
import wandb
from tqdm import tqdm
from utils.utils import *
# from vae.vae import GMVAE
import torch.optim as optim
from dqn.dqn import DQNAgent
from sac_agent.agent import SAC
from partedvae.models import VAE
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
# from utils.get_llm_output import GetLLMGoals
# from sentence_transformers import SentenceTransformer
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
    imagination_net = ImaginationNet(env = env,
                                     config = config,
                                     num_goals = num_goals,
                                     vae = vae,
                                     agent = agent).to(device)
    wandb.watch(imagination_net)
    #Creating the optimizer
    optimizer = optim.Adam(imagination_net.parameters(), lr=config['lr'])
    # nn_utils.clip_grad_norm_(imagination_net.parameters(), max_norm=config['max_gradient'])
    epoch_bar = tqdm(range(config.Network.epoch), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0
        total_class_loss = 0
        total_policy_loss = 0
        total_proximity_loss = 0
        for data, indices in dataloader:
            data = data.float().to(device)
            optimizer.zero_grad()
            imagined_state = imagination_net(data)
            loss, class_loss, policy_loss, proximity_loss  = imagination_net.compute_loss(data, imagined_state)
            loss.backward()
            optimizer.step()
            # Accumulate losses
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_policy_loss += policy_loss.item()
            total_proximity_loss += proximity_loss.item()
            epoch_bar.set_postfix({'Loss': loss.item(), 'Class loss': class_loss.item(), 'Policy loss': policy_loss.item(), 'Proximity loss': proximity_loss.item()})
            
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
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'imagination_net_epoch_{epoch}.tar')
            imagination_net.save(checkpoint_path)
        # print(f"Epoch {epoch}/{epochs} - Loss: {average_loss:.4f}, Recon Loss: {average_recon:.4f}, KL Divergence: {average_kl:.4f}, KL Weight: {kl_weight:.4f}")
    imagination_net.save('models/imagination_net.pth')
    print("Training complete.")

@hydra.main(version_base=None, config_path="config", config_name="imagination_net_master_config")
def main(args: DictConfig) -> None:
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    #Loading the dataset
    dataset = get_data(f'{args.General.datapath}/{args.General.env}/data.pkl')
    # captions = get_data(f'{args.General.datapath}/{args.General.encoder_model}/captions.pkl')
    # Initialize dataset and dataloader
    dataset = TextDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=args.Network.batch_size, shuffle=True)
    if args.General.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.General.max_ep_len, agent_view_size=5, size=7)
    # dataset = normalize_data(dataset)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    # goal_gen = GetLLMGoals()
    # goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    # sentencebert = SentenceTransformer(args.General.encoder_model)
    # for params in sentencebert.parameters():
    #     params.requires_grad = False
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    # with torch.no_grad():
    #     mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device, show_progress_bar=False)
    # Define prior means (mu_p) for each mixture component
    # Initialize VAE with learnable prior means
    # latent_dim = sentencebert.get_sentence_embedding_dimension()
    # num_mixtures = len(goals)
    # vae = GMVAE(
    #     input_dim = dataset[0].shape[0], 
    #     encoder_hidden = args.Network.encoder_hidden,
    #     decoder_hidden = args.Network.decoder_hidden, 
    #     latent_dim=latent_dim, 
    #     num_mixtures=num_mixtures, 
    #     mu_p=mu_p
    # )
    #Loading the pretrained VAE
    # vae.load(args.General.vae_checkpoint)
    # vae.to(device)
    #Freezing the VAE weight to prevent updating
    # for params in vae.parameters():
    #     params.requires_grad = False
    
    #Loading Parted VAE
    latent_spec = args.Training.latent_spec
    disc_priors = [[1/args.General.num_goals]*args.General.num_goals]
    
    vae = VAE(args.Training.input_dim, 
              args.Training.encoder_hidden_dim, 
              args.Training.decoder_hidden_dim, 
              args.Training.output_size, 
              latent_spec, 
              c_priors=disc_priors, 
              save_dir='',
              device=device)
    vae.load(args.General.vae_checkpoint)
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
    train_imagination_net(config = args, 
                          env= env,
                          vae = vae,
                          agent = agent, 
                          dataloader = dataloader, 
                          checkpoint_interval = args.Network.checkpoint_interval, 
                          checkpoint_dir = model_dir,
                          num_goals = args.General.num_goals)
    
if __name__ == "__main__":
    wandb.init(project="Imagination-net_training")
    main()
    