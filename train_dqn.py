import hydra
import wandb
import warnings
from dqn.dqn import DQNAgent
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="config", config_name="dqn")
def main(args: DictConfig) -> None:
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    args_gen = args.General
    policy_config = args.policy_config
    policy_network_cfg = args.policy_network_cfg
    policy_optim_cfg = args.policy_network_cfg
    
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    
    save_path = "models/dqn_agent"
    if args_gen.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args_gen.max_ep_len, agent_view_size=5, size=7)
    dqn_agent = DQNAgent(env, args_gen, policy_config, policy_network_cfg, policy_optim_cfg, save_path)
    dqn_agent.learn()
    
if __name__ == "__main__":
    wandb.init(project="Imagination_DQN_training")
    main()