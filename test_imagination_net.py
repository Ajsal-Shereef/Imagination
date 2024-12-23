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
from utils.utils import write_video
# from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
from utils.get_llm_output import GetLLMGoals
from sentence_transformers import SentenceTransformer
from imagination.imagination_net import ImaginationNet
from architectures.m2_vae.dgm import DeepGenerativeModel

is_agent = False

@hydra.main(version_base=None, config_path="config", config_name="master_config")
def main(args: DictConfig) -> None:
    if args.General.env ==  "SimplePickup":
        from env.env import SimplePickup, generate_caption, MiniGridTransitionDescriber, calculate_probabilities
        env = SimplePickup(max_steps=args.General.max_ep_len, agent_view_size=5, size=7, render_mode="rgb_array")
        
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
                    state_size = env.observation_space.shape[0],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.Imagination_General.buffer_size)
    #Loading the pretrained weight of sac agent.
    agent.load_params(args.Imagination_General.agent_checkpoint)
    #Freezing the SAC agent weight to prevent updating
    for params in agent.parameters():
        params.requires_grad = False
        
    vae = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.num_goals, args.M2_Network.latent_dim, args.M2_Network.encoder_hidden_dim]).to(device)
    vae.load(args.Imagination_General.vae_checkpoint)
    vae.to(device)
    vae.eval()
    
    for params in vae.parameters():
        params.requires_grad = False
        
    imagination_net = ImaginationNet(env = env,
                                     config = args,
                                     num_goals = args.Imagination_General.num_goals,
                                     agent = agent,
                                     vae = vae).to(device)
        
    imagination_net.load("models/imagination_net/imagination_net_epoch_800.tar")
    imagination_net.eval()
    
    transition_captioner = MiniGridTransitionDescriber(5)
        
    # Create data directory if it doesn't exist
    if is_agent:
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
        hx = None
        while True:
            env.render()
            p_agent_loc = env.agent_pos
            p_state = env.get_unprocesed_obs()
            if is_agent:
                with torch.no_grad():
                    action = agent.get_action(state)
            else:
                with torch.no_grad():
                    caption = generate_caption(state[:-4].reshape((5,5,3)))
                    imagined_state = imagination_net(torch.tensor(state).float().to(device))
                    action = agent.get_action(imagined_state[0,0,:])
                    original_state_prob = calculate_probabilities(env.agent_pos,env.get_unprocesed_obs()['image'],env.get_unprocesed_obs()['direction'], env.purple_key_loc, 
                                           env.green_ball_loc)
                    _, _, _, _, vae_original_prob = vae(torch.tensor(state).to(device).float())
                    
                    arr1 = np.array([2., 5., 0., 2., 5., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 2., 5., 0., 2., 5., 0., 1., 0., 0.,
                                     1., 0., 0., 1., 0., 0., 2., 5., 0., 2., 5., 0., 6., 1., 0., 1., 0., 0., 1., 0., 0., 2., 5., 0.,
                                     2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0.,
                                     2., 5., 0., 0., 1., 0., 0.,])
                    
                    arr2 = np.array([2., 5., 0., 2., 5., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 2., 5., 0., 2., 5., 0., 1., 0., 0.,
                                     1., 0., 0., 1., 0., 0., 2., 5., 0., 2., 5., 0., 5., 3., 0., 1., 0., 0., 1., 0., 0., 2., 5., 0.,
                                     2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0.,
                                     2., 5., 0., 0., 1., 0., 0.,])
                    
                    test1 = vae(torch.tensor(arr1).to(device).float(), torch.tensor([1,0]).to(device).float())
                    test11 = vae(test1[0], torch.tensor([1,0]).to(device).float())
                    test2 = vae(torch.tensor([2., 5., 0., 2., 5., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 2., 5., 0., 2., 5., 0., 1., 0., 0.,
                                              1., 0., 0., 1., 0., 0., 2., 5., 0., 2., 5., 0., 6., 1., 0., 1., 0., 0., 1., 0., 0., 2., 5., 0.,
                                              2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0., 2., 5., 0.,
                                              2., 5., 0., 0., 1., 0., 0.]).to(device).float())
                    _, _, _, _, vae_1 = vae(imagined_state[0,1,:])
                    _, _, _, _, vae_0 = vae(imagined_state[0,0,:])
                    cv2.imwrite("frame.png", frame)
            next_state, reward, terminated, truncated, _ = env.step(action)
            c_agent_loc = env.agent_pos
            c_state = env.get_unprocesed_obs()
            transition_caption = transition_captioner.generate_description(agent_prev_pos = p_agent_loc, 
                                                                           agent_curr_pos = c_agent_loc, 
                                                                           agent_prev_dir = p_state['direction'], 
                                                                           agent_curr_dir = c_state['direction'], 
                                                                           prev_view = p_state['image'],
                                                                           curr_view = c_state['image'],
                                                                           purple_key_pos = env.purple_key_loc, 
                                                                           green_ball_pos = env.green_ball_loc,
                                                                           agent_action = action)
            
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