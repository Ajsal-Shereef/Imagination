import os
import cv2
import hydra
import torch
import numpy as np
import gymnasium as gym

from env.env import reverse_preprocess_observation
from vae.vae import GMVAE
from sac_agent.agent import SAC
from utils.utils import write_video
# from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
# from utils.get_llm_output import GetLLMGoals
# from sentence_transformers import SentenceTransformer
from imagination.imagination_net import ImaginationNet

is_sac_agent = True

@hydra.main(version_base=None, config_path="config", config_name="imagination_net")
def main(args: DictConfig) -> None:
    if args.General.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.General.max_ep_len, agent_view_size=5, size=7, render_mode="rgb_array")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get the goals from the LLM. #TODO Need to supply the controllable entity within the environment
    # goal_gen = GetLLMGoals()
    # goals = goal_gen.llmGoals([])
    # Load the sentecebert model to get the embedding of the goals from the LLM
    # sentencebert = SentenceTransformer(args.General.encoder_model)
    # Define prior means (mu_p) for each mixture component as the output from the sentencebert model
    # mu_p = sentencebert.encode(goals, convert_to_tensor=True, device=device)
    # Define prior means (mu_p) for each mixture component
    # Initialize VAE with learnable prior means
    mu_p = torch.randn(2, 384)
    # latent_dim = sentencebert.get_sentence_embedding_dimension()
    latent_dim = 384
    num_mixtures = 2
    vae = GMVAE(
        input_dim = 504, 
        encoder_hidden = [1024,1024,512,512,512,256,256,256,256], #Don't forget to edit the snippet above as well
        decoder_hidden = [256,256], 
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
    
    sac_agent = SAC(args,
                    state_size = env.observation_space.shape[0],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.General.buffer_size)
    #Loading the pretrained weight of sac agent.
    sac_agent.load_params(args.General.sac_agent_checkpoint)
    #Freezing the SAC agent weight to prevent updating
    for params in sac_agent.parameters():
        params.requires_grad = False
        
    imagination_net = ImaginationNet(input_dim = 504,
                                     hidden_layers = [256, 512, 512],
                                     num_goals = num_mixtures,
                                     vae = vae,
                                     sac = sac_agent).to(device)
    imagination_net.load("models/imagination_net/imagination_net_epoch_20000.pth")
        
    # Create data directory if it doesn't exist
    if is_sac_agent:
        video_dir = "Videos/Original"
    else:
        video_dir = "Videos/Imagination"
    
    for i in range(1, 10):
        frame_arrays = []
        state, info = env.reset()
        frame = env.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_arrays.append(frame)
        episode_steps = 0
        while True:
            env.render()
            if is_sac_agent:
                with torch.no_grad():
                    action = sac_agent.get_action(state)
            else:
                with torch.no_grad():
                    imagined_state = imagination_net(state)
                imagined_state = imagined_state.squeeze(0).detach().cpu().numpy()
                r0 = reverse_preprocess_observation(np.round(imagined_state[0]).astype(int),5,5)
                r1 = reverse_preprocess_observation(np.round(imagined_state[1]).astype(int),5,5)
                difference0 = np.argmax(np.abs(imagined_state[0] - state))
                difference1 = np.argmax(np.abs(imagined_state[1] - state))
                diff_value0 = imagined_state[0][np.argmax(np.abs(imagined_state[0] - state))]
                diff_value1 = imagined_state[1][np.argmax(np.abs(imagined_state[1] - state))]
                action = sac_agent.get_action(imagined_state[0])
            next_state, reward, terminated, truncated, _ = env.step(action)
            frame = env.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_arrays.append(frame)
            done = terminated + truncated
            state = next_state
            episode_steps += 1
            if done:
                break
        write_video(frame_arrays, str(i), video_dir)    
if __name__ == "__main__":
    main()