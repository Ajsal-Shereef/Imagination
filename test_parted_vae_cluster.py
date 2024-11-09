import os
import cv2
import hydra
import torch
import warnings
from partedvae.models import VAE
from env.env import SimplePickup, generate_caption, calculate_probabilities
from omegaconf import DictConfig, OmegaConf

vae_path = "models/parted_vae/parted_vae_37100.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

@hydra.main(version_base=None, config_path="config", config_name="parted_vae")
def main(args: DictConfig) -> None:
    env = SimplePickup(max_steps=20, agent_view_size=5, size=7)
    
    latent_spec = args.Training.latent_spec
    disc_priors = [[1/args.General.num_goals]*args.General.num_goals]
    save_dir = os.makedirs(f'{args.General.load_model_path}/parted_vae', exist_ok=True)
    
    model = VAE(args.Training.input_dim, 
                args.Training.encoder_hidden_dim, 
                args.Training.decoder_hidden_dim, 
                args.Training.output_size, 
                latent_spec, 
                c_priors=disc_priors, 
                save_dir=save_dir,
                device=device)
    
    model.load("models/parted_vae/parted_vae_55000.pth")
    model.to(device)
    model.eval()
    total_step = 0
    while total_step < 20000:
        state, info = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_step += 1
            vae_input = torch.tensor(next_state).unsqueeze(0).float().to(device)
            with torch.no_grad():
                reconstruction, inference_out = model(vae_input)
            cluster_prob = torch.exp(inference_out['log_c'])
            frame = env.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cluster_prob[0][0] > 0.85:
                prob = calculate_probabilities(env.agent_pos, 
                                       env.get_unprocesed_obs()['image'], 
                                       env.get_unprocesed_obs()['direction'], 
                                       env.red_ball_loc, 
                                       env.green_ball_loc)
                print("First gaussian is greater than 85% " + generate_caption(env.get_unprocesed_obs()['image'])[1])
            elif cluster_prob[0][1] > 0.85:
                prob = calculate_probabilities(env.agent_pos, 
                                       env.get_unprocesed_obs()['image'], 
                                       env.get_unprocesed_obs()['direction'], 
                                       env.red_ball_loc, 
                                       env.green_ball_loc)
                print("Second gaussian is greater than 85% " + generate_caption(env.get_unprocesed_obs()['image'])[1])
            elif cluster_prob[0][1] > 0.40 and cluster_prob[0][1] < 0.60:
                print("Second gaussian is between 40 and 60 " + generate_caption(env.get_unprocesed_obs()['image'])[1])
            elif cluster_prob[0][0] > 0.40 and cluster_prob[0][0] < 0.60:
                print("First gaussian is between 40 and 60 " + generate_caption(env.get_unprocesed_obs()['image'])[1])
            done = terminated + truncated
            state = next_state
            if done:
                break
            
if __name__ == "__main__":
    main()

