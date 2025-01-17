import os
import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from helper_functions.utils import *
from sac_agent.agent import SAC
from partedvae.models import VAE
from helper_functions.visualize import Visualizer
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from imagination.imagination_net import ImaginationNet
from architectures.m2_vae.dgm import DeepGenerativeModel
from helper_functions.collect_vae_training_data import collect_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "models/m2_vae/2025-01-15_14-08-34_B6SHS9/model.pt"

def visualize_latent_space(model, imagination_net, dataloader, device, all_z=[], all_labels=[], method='pca', save_path=''):
    """
    Visualize the latent space using PCA or t-SNE.

    Args:
        model (VAE): Trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        method (str): 'pca' or 'tsne'.
        save_path (str): Path to save the visualization plot.
    """
    model.eval()
    if len(all_z) == 0:
        all_z = []
        all_labels = []
        true_labels = []
        imagined_state_0_pred = []
        imagined_state_1_pred = []
        imagined_z_0 = []
        imagined_z_1 = []
        with torch.no_grad():
            for data, true_label in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                data = data.float().to(device)
                
                #Invoking imagined_state
                # imagined_state = imagination_net(data)
                # _, img_0_z, _, _, imagined_state_0_label = model(imagined_state[:,0,:])
                # _, img_1_z, _, _, imagined_state_1_label = model(imagined_state[:,1,:])
                # imagined_z_0.append(img_0_z.detach().cpu().numpy())
                # imagined_z_1.append(img_1_z.detach().cpu().numpy())
                # imagined_state_0_pred.append(imagined_state_0_label.detach().cpu().numpy())
                # imagined_state_1_pred.append(imagined_state_1_label.detach().cpu().numpy())
                
                # frame = cv2.imread('frame.png')
                # frame = np.load('frame.npy')
                # frame = np.transpose(frame, (2, 0, 1))/255
                # test = model(torch.tensor(frame).to(device).float())
                # test0 = model(torch.tensor(frame).to(device).float(), test[-1])
                # test1 = model(torch.tensor(frame).to(device).float(), torch.flip(test[-1], dims=[-1]))
                # class_0 = model(test0[0])
                # class_1 = model(test1[0])
                # generated = np.transpose((test0[0]*255).squeeze().detach().cpu().numpy(), (1, 2, 0))
                # generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(generated.astype(np.uint8))
                # image.save("gen.png")
                # plt.imshow(generated.clip(0, 1))  # Clipping is redundant if already in [0, 1]
                # plt.axis('off')  # Remove axes for visualization
                # plt.savefig('plot.png')
                # generated1 = np.transpose((test1[0]*255).squeeze().detach().cpu().numpy(), (1, 2, 0))
                # np.save("gen1.npy", generated1)
                # image = Image.fromarray(generated1.astype(np.uint8))
                # image.save("gen1.png")
                # generated1 = cv2.cvtColor(generated1, cv2.COLOR_BGR2RGB)
                # plt.imshow(generated1.clip(0, 1))  # Clipping is redundant if already in [0, 1]
                # plt.axis('off')  # Remove axes for visualization
                # plt.show()
                
                # print(f'gen1 {model(test1[0])[-1]}')
                
                # test_11 = model(test1[0], torch.flip(test[-1], dims=[-1]))
                # generated11 = np.transpose(test_11[0].squeeze().detach().cpu().numpy(), (1, 2, 0))
                # generated11 = cv2.cvtColor(generated11, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("gen11.png", generated11*255)
                
                # print(f'gen11 {model(test_11[0])[-1]}')
                
                # test_111 = model(test_11[0], torch.flip(test[-1], dims=[-1]))
                # generated111 = np.transpose(test_111[0].squeeze().detach().cpu().numpy(), (1, 2, 0))
                # generated111 = cv2.cvtColor(generated111, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("gen111.png", generated111*255)
                
                # print(f'gen111 {model(test_111[0])[-1]}')
                
                # test_1111 = model(test_111[0], torch.flip(test[-1], dims=[-1]))
                # generated1111 = np.transpose(test_1111[0].squeeze().detach().cpu().numpy(), (1, 2, 0))
                # generated1111 = cv2.cvtColor(generated1111, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("gen1111.png", generated1111*255)
                
                # print(f'gen1111 {model(test_1111[0])[-1]}')
                
                # test_11111 = model(test_1111[0], torch.flip(test[-1], dims=[-1]))
                # generated11111 = np.transpose(test_11111[0].squeeze().detach().cpu().numpy(), (1, 2, 0))
                # generated11111 = cv2.cvtColor(generated11111, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("gen11111.png", generated11111*255)
                
                # print(f'gen11111 {model(test_11111[0])[-1]}')
                
                
                true_label = true_label.float().to(device)
                true_labels.append(torch.argmax(true_label, dim=-1).detach().cpu().numpy())
                # Forward pass
                x_reconstructed, x_z, x_z_mu, x_z_log_var, u_c_logits, u_c = model(data)
                latent = x_z
                all_z.append(latent.detach().cpu().numpy())
                all_labels.append(torch.argmax(u_c_logits, dim=-1).detach().cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        # imagined_state_0_pred = np.concatenate(imagined_state_0_pred, axis=0)
        # imagined_state_1_pred = np.concatenate(imagined_state_1_pred, axis=0)
        # imagined_z_0 = np.concatenate(imagined_z_0, axis=0)
        # imagined_z_1 = np.concatenate(imagined_z_1, axis=0)
        
    fig_save_name = f'{save_path}/latent_space_.png'
    true_fig_save_name = f'{save_path}/latent_space_true_label.png'

    # plt.scatter(all_z[:,0], all_z[:,1], c=all_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    # plt.colorbar(label='Probability of Class 1')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Class latent")
    # plt.savefig(fig_save_name)
    # plt.close()
    
    # plt.scatter(all_z[:,0], all_z[:,1], c=true_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    # plt.colorbar(label='True probability of Class 1')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Class latent")
    # plt.savefig(true_fig_save_name)
    # plt.close()
    
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_pca_.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_tsne.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    
    unique_labels = np.unique(all_labels)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        # Filter points by label
        mask = all_labels == label
        plt.scatter(reduced_z[:,0][mask], reduced_z[:,1][mask], label=f'Class {label}', alpha=0.7)
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.legend()
    plt.savefig(fig_save_name)
    plt.close()
    
    unique_labels = np.unique(true_labels)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        # Filter points by label
        mask = all_labels == label
        plt.scatter(reduced_z[:,0][mask], reduced_z[:,1][mask], label=f'Class {label}', alpha=0.7)
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.legend()
    plt.savefig(true_fig_save_name)
    plt.close()
    
    
    # if method == 'pca':
    #     reducer = PCA(n_components=2)
    #     reduced_z = reducer.fit_transform(imagined_z_0)
    #     fig_save_name = f'{save_path}/latent_space_pca_imagined_0.png'
    # elif method == 'tsne':
    #     reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #     reduced_z = reducer.fit_transform(all_z)
    #     fig_save_name = f'{save_path}/latent_space_tsne.png'
    # else:
    #     raise ValueError("Method must be 'pca' or 'tsne'.")
    
    # plt.scatter(reduced_z[:,0], reduced_z[:,1], c=imagined_state_0_pred[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    # plt.colorbar(label='Predicted probability of Class 1')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Class latent")
    # plt.savefig(fig_save_name)
    # plt.close()
    
    # if method == 'pca':
    #     reducer = PCA(n_components=2)
    #     reduced_z = reducer.fit_transform(imagined_z_1)
    #     fig_save_name = f'{save_path}/latent_space_pca_imagined_1.png'
    # elif method == 'tsne':
    #     reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    #     reduced_z = reducer.fit_transform(all_z)
    #     fig_save_name = f'{save_path}/latent_space_tsne.png'
    # else:
    #     raise ValueError("Method must be 'pca' or 'tsne'.")
    
    # plt.scatter(reduced_z[:,0], reduced_z[:,1], c=imagined_state_1_pred[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    # plt.colorbar(label='Predicted probability of Class 1')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Class latent")
    # plt.savefig(fig_save_name)
    # plt.close()
    
    
    print(f"Latent space visualization saved to {fig_save_name}")
    return all_z, all_labels

@hydra.main(version_base=None, config_path="config", config_name="master_config")
def main(args: DictConfig) -> None:
    #Loading the dataset
    if args.M2_General.env == "SimplePickup":
        from env.env import SimplePickup #TransitionCaptioner
        env = SimplePickup(max_steps=20, agent_view_size=5, size=7)
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env)
        
    # datasets, _, class_probs = collect_data(env, True, 2000, 20, device)
    
    #Loading the dataset
    datasets = get_data(f'{args.M2_General.datapath}/{args.M2_General.env}/data.pkl')
    datasets = [i/255 for i in datasets]
    # captions = get_data(f'{args.General.datapath}/{args.General.env}/captions.pkl')
    class_probs = get_data(f'{args.M2_General.datapath}/{args.M2_General.env}/class_prob.pkl')
    
    # Initialize dataset and dataloader
    dataloader = TwoListDataset(datasets, class_probs)
    train_loader = DataLoader(dataloader, batch_size=900, shuffle=True)
    
    if args.Imagination_General.agent == 'dqn':
        agent = DQNAgent(env, 
                        args.General, 
                        args.policy_config, 
                        args.policy_network_cfg, 
                        args.policy_network_cfg, '')
    else:
        agent = SAC(args,
                    input_dim = env.observation_space['image'].shape[-1],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.Imagination_General.buffer_size)
        
    #Loading the pretrained weight of sac agent.
    # agent.load_params(args.Imagination_General.agent_checkpoint)
    
    model = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.y_dim, args.M2_Network.h_dim, \
                                 args.M2_Network.latent_dim, args.M2_Network.classifier_hidden_dim, args.M2_Network.feature_encoder_channel_dim], \
                                 args.M2_Network.label_loss_weight,
                                 args.M2_Network.recon_loss_weight).to(device)
    model.load(model_dir)
    model.to(device)
    model.eval()
    
    imagination_net = ImaginationNet(env = env,
                                     config = args,
                                     num_goals = args.Imagination_General.num_goals,
                                     agent = agent,
                                     vae = model).to(device)
        
    # imagination_net.load("models/imagination_net/imagination_net_epoch_1000.tar")
    # imagination_net.eval()
    
    # Visualization of the latent space 
    data_dir = f'visualizations/m2_vae'
    viz = Visualizer(model, device, root=f'{data_dir}/')
    os.makedirs(data_dir, exist_ok=True)
    latent, labels = visualize_latent_space(model, imagination_net, train_loader, device, method='pca', save_path=data_dir)
    
    for batch, labels in train_loader:
        break
    viz.reconstructions(data=batch, label=labels.to(device))
    # latent = visualize_latent_space(model, dataloader, device, latent, labels, method='tsne', save_path=data_dir)
    
if __name__ == "__main__":
    main()