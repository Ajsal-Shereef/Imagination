
import cv2
import hydra
import torch
import numpy as np
from architectures.m2_vae.dgm import DeepGenerativeModel
from omegaconf import DictConfig, OmegaConf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "models/m2_vae/2025-01-14_13-02-43_TSGVV1/model.pt"

@hydra.main(version_base=None, config_path="config", config_name="master_config")
def main(args: DictConfig) -> None:
    model = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.y_dim, args.M2_Network.h_dim, \
                                 args.M2_Network.latent_dim, args.M2_Network.classifier_hidden_dim, args.M2_Network.feature_encoder_channel_dim], \
                                 args.M2_Network.label_loss_weight).to(device)
    model.load(model_dir)
    model.to(device)
    model.eval()

    frame = cv2.imread("frame.png")

    frame = np.transpose(frame, (2, 0, 1))/255

    model_output = model(torch.tensor(frame).to(device).float())
    print("Model output")

if __name__ == "__main__":
    main()

