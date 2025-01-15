import torch
import wandb
import argparse
import gymnasium
import numpy as np
from collections import deque
from helper_functions.utils import *
import random
from sac_agent.agent import SAC
from sac_agent.buffer import StateCaptionEncodeBuffer
from sentence_transformers import SentenceTransformer

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="Training SAC agents", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="SimplePickup", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes, default: 100")
    parser.add_argument("--max_ep_len", type=int, default=20, help="Number of timestep within an episode")
    parser.add_argument("--buffer_size", type=int, default=1000_00, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=5000, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L12-v2", help="Sentence encoder")
    args = parser.parse_args()
    return args

def train(config):
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    if config.env ==  "SimplePickup":
        from env.env import SimplePickup, MiniGridTransitionDescriber
        env = SimplePickup(max_steps=config.max_ep_len, agent_view_size=5, size=7)
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average = deque(maxlen=100)
    total_steps = 0
    
    with wandb.init(project="SAC_Discrete", name=config.run_name, config=config):
        
        sac_agent = SAC(config,
                        input_dim = env.observation_space['image'].shape[-1],
                        action_size = env.action_space.n,
                        device=device,
                        buffer_size = config.buffer_size)
        
        # wandb.watch(sac_agent, log="gradients", log_freq=100)
        
        # transition_captioner = MiniGridTransitionDescriber(5)
        # sentencebert = SentenceTransformer(config.encoder)
        # statecaption_buffer = StateCaptionEncodeBuffer(buffer_size = config.buffer_size, device=device)
                                                     
        
        collect_random(env=env, dataset=sac_agent.buffer, num_samples=2000)
        for i in range(1, config.episodes+1):
            state, info = env.reset()
            state = np.transpose(state['image'], (2, 0, 1))/255
            # c_agent_loc = env.agent_pos
            # i = env.get_unprocesed_obs()['direction']
            # transition_caption = transition_captioner.generate_description(agent_prev_pos = None, 
            #                                                                agent_curr_pos = c_agent_loc, 
            #                                                                agent_prev_dir = None, 
            #                                                                agent_curr_dir = env.get_unprocesed_obs()['direction'], 
            #                                                                prev_view = None,
            #                                                                curr_view = env.get_unprocesed_obs()['image'],
            #                                                                purple_key_pos = env.purple_key_loc, 
            #                                                                green_ball_pos = env.green_ball_loc,
            #                                                                agent_action = None)
            # caption_encoding = sentencebert.encode(transition_caption, convert_to_tensor=True, device=device)
            episode_steps = 0
            rewards = 0
            while True:
                action = sac_agent.get_action(state)
                steps += 1
                # p_state = env.get_unprocesed_obs()
                # p_agent_loc = env.agent_pos
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.transpose(next_state['image'], (2, 0, 1))/255
                # c_state = env.get_unprocesed_obs()
                # c_agent_loc = env.agent_pos
                # transition_caption = transition_captioner.generate_description(agent_prev_pos = p_agent_loc, 
                #                                                                agent_curr_pos = c_agent_loc, 
                #                                                                agent_prev_dir = p_state['direction'], 
                #                                                                agent_curr_dir = c_state['direction'], 
                #                                                                prev_view = p_state['image'],
                #                                                                curr_view = c_state['image'],
                #                                                                purple_key_pos = env.purple_key_loc, 
                #                                                                green_ball_pos = env.green_ball_loc,
                #                                                                agent_action = action)
                # c_frame = env.get_frame()
                # c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("frame.png", c_frame)
                # caption_encoding = sentencebert.encode(transition_caption, convert_to_tensor=True, device=device)
                done = terminated + truncated
                sac_agent.buffer.add(state, action, reward, next_state, done)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = sac_agent.learn(steps, 
                                                                                                           sac_agent.buffer.sample(),
                                                                                                           gamma=0.99)
                # statecaption_buffer.add(state, next_state, caption_encoding)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            average.append(rewards)
            total_steps += episode_steps
            #print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            wandb.log({"Reward": rewards,
                       "Average": np.mean(average),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Steps": steps,
                       "Episode": i, 
                       "Buffer size": sac_agent.buffer.__len__()})

            # if (i %10 == 0) and config.log_video:
            #     mp4list = glob.glob('video/*.mp4')
            #     if len(mp4list) > 1:
            #         mp4 = mp4list[-2]
            #         wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                sac_agent.save("models/sac_agent", save_name="SAC_discrete")
        #Saving the model
        sac_agent.save("models/sac_agent", save_name="SAC_discrete")
        #Saving the trajectory data
        # statecaption_buffer.dump_buffer_data(f"data/{config.env}")       
        #Testing the training
        test(env, sac_agent, "Videos/Sac_agent", n_episode=20)
            
if __name__ == "__main__":
    config = get_config()
    train(config)
