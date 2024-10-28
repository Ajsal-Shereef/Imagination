import numpy as np
import torch
import torch.nn as nn 

class NeuralNet(nn.Module):
    def __init__(self, input_size, 
                 hidder_layer, num_actions):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidder_layer)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidder_layer, num_actions)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
       hidder_layer_out =  self.l1(x)
       relu_out = self.relu(hidder_layer_out)
       final_layer_out = self.l2(relu_out)
       final_out = self.softmax(final_layer_out)
       return final_out
   
   