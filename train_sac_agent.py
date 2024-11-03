import torch
import wandb
import argparse
import gymnasium
import numpy as np
from collections import deque
from utils.utils import *
import random
from sac_agent.agent import SAC

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="Training SAC agents", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="SimplePickup", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes, default: 100")
    parser.add_argument("--max_ep_len", type=int, default=20, help="Number of timestep within an episode")
    parser.add_argument("--buffer_size", type=int, default=3000_00, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=5000, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args

def train(config):
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    if config.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=config.max_ep_len, agent_view_size=5, size=7)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average = deque(maxlen=100)
    total_steps = 0
    
    with wandb.init(project="SAC_Discrete", name=config.run_name, config=config):
        
        sac_agent = SAC(config,
                        state_size = env.observation_space.shape[0],
                        action_size = env.action_space.n,
                        device=device,
                        buffer_size = config.buffer_size)
        
        wandb.watch(sac_agent, log="gradients", log_freq=100)
        
        collect_random(env=env, dataset=sac_agent.buffer, num_samples=2000)
        
        for i in range(1, config.episodes+1):
            state, info = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = sac_agent.get_action(state)
                steps += 1
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated + truncated
                sac_agent.buffer.add(state, action, reward, next_state, done)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = sac_agent.learn(steps, 
                                                                                                           sac_agent.buffer.sample(),
                                                                                                           gamma=0.99)
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
        sac_agent.save("models/sac_agent", save_name="SAC_discrete")            
        #Testing the training
        test(env, sac_agent, "Videos/Sac_agent", n_episode=20)
            
if __name__ == "__main__":
    config = get_config()
    train(config)
