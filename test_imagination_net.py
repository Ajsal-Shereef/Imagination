import os
import cv2
import hydra
import torch
import numpy as np
import gymnasium as gym
from partedvae.models import VAE

from env.env import reverse_preprocess_observation
from vae.vae import GMVAE
from sac_agent.agent import SAC
from dqn.dqn import DQNAgent
from helper_functions.utils import write_video
# from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
from helper_functions.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer
from imagination.imagination_net import ImaginationNet
from architectures.m2_vae.dgm import DeepGenerativeModel

is_agent = False

@hydra.main(version_base=None, config_path="config", config_name="master_config")
def main(args: DictConfig) -> None:
    if args.General.env ==  "SimplePickup":
        from env.env import MultiObjectMiniGridEnv
        env = MultiObjectMiniGridEnv(max_steps=args.General.max_ep_len, agent_view_size=5, size=7, render_mode="rgb_array")
        # from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        # env = RGBImgPartialObsWrapper(env)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    # goal_gen = GetLLMGoals()
    # goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    # sentencebert = SentenceTransformer(args.Imagination_General.encoder_model)
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    # mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    # Define prior means (mu_p) for each mixture component
    # Initialize VAE with learnable prior means
    # mu_p = torch.randn(2, 384)
    # latent_dim = sentencebert.get_sentence_embedding_dimension()
    # latent_dim = 384
    # num_mixtures = 2
    # vae = GMVAE(
    #     input_dim = 504, 
    #     encoder_hidden = [1024,1024,512,512,512,256,256,256,256], #Don't forget to edit the snippet above as well
    #     decoder_hidden = [256,256], 
    #     latent_dim=latent_dim, 
    #     num_mixtures=num_mixtures, 
    #     mu_p=mu_p
    # )
    # #Loading the pretrained VAE
    # vae.load(args.General.vae_checkpoint)
    # vae.to(device)
    
    #Freezing the VAE weight to prevent updating
    # for params in vae.parameters():
    #     params.requires_grad = False
    
    if args.Imagination_General.agent == 'dqn':
        agent = DQNAgent(env, 
                        args.General, 
                        args.policy_config, 
                        args.policy_network_cfg, 
                        args.policy_network_cfg, '')
    else:
        agent = SAC(args,
                    input_dim = env.observation_space.shape[-1],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.Imagination_General.buffer_size)
    #Loading the pretrained weight of sac agent.
    agent.load_params(args.Imagination_General.agent_checkpoint)
    agent.eval()
    #Freezing the SAC agent weight to prevent updating
    for params in agent.parameters():
        params.requires_grad = False
        
    model = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.y_dim, args.M2_Network.h_dim, \
                                 args.M2_Network.latent_dim, args.M2_Network.classifier_hidden_dim, \
                                 args.M2_Network.feature_encoder_hidden], \
                                 args.M2_Network.label_loss_weight,
                                 args.M2_Network.recon_loss_weight).to(device)
    model.load(args.Imagination_General.vae_checkpoint)
    model.to(device)
    model.eval()
    
    for params in model.parameters():
        params.requires_grad = False
        
    # Create data directory if it doesn't exist
    if is_agent:
        video_dir = "Videos/Original"
    else:
        video_dir = "Videos/Imagination"
    
    for i in range(1, 10):
        frame_arrays = []
        partial_view_array = []
        state, info = env.reset()
        frame = env.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_arrays.append(frame)
        episode_steps = 0
        while True:
            env.render()
            p_agent_loc = env.agent_pos
            # p_state = env.get_unprocesed_obs()
            if is_agent:
                with torch.no_grad():
                    action = agent.get_action(state)
            else:
                with torch.no_grad():
                    x = torch.tensor(state).to(device).float()
                    # imagined_state = model.generate(x.unsqueeze(0), torch.tensor([0, 1, 0]).unsqueeze(0).to(device).float())
                    # imagined_state = model(x.unsqueeze(0))
                    # imagined_state  = imagined_state[0]
                    action = agent.get_action(imagined_state.squeeze())
                    imagined_state = np.round(imagined_state.squeeze().detach().cpu().numpy())
                    r_obs_gen = np.clip(imagined_state[:-10], 0, 5).reshape((5,5,3))
                    partial_view_gen = env.render_partial_view_from_features(r_obs_gen, np.argmax(imagined_state[-10:-6]), tile_size=32)
                    r_obs = state[:-10].reshape((5,5,3))
                    partial_view = env.render_partial_view_from_features(r_obs, np.argmax(state[-10:-6]), tile_size=32)
                    # Concatenate tensors side by side (along the width)
                    concatenated_image = np.concatenate((partial_view_gen, partial_view), axis=1)  # Shape: (3, 40, 80)
                    # Convert PyTorch tensor to NumPy array for OpenCV (HWC format)
                    # partial_frame = (concatenated_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  # (40, 80, 3)
                    partial_view_array.append(cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2RGB))

            next_state, reward, terminated, truncated, _ = env.step(action)
            frame = env.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_arrays.append(frame)
            done = terminated + truncated
            state = next_state
            episode_steps += 1
            if done:
                break
        write_video(frame_arrays, f'{str(i)}_full_view', f'{video_dir}/Full_view')
        write_video(partial_view_array, f'{str(i)}_partial_view', f'{video_dir}/Partial_view', (320,160))    
if __name__ == "__main__":
    main()
    