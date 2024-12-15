from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InstanceEncoder(nn.Module):
    def __init__(self):
        super(InstanceEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 96, 96)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, 48, 48)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 48, 48)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 24, 24)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, 12, 12)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (B, 256, 6, 6)
        )

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)  # (B, 256, 6, 6)
        x = x.view(x.size(0), -1)  # (B, 256 * 6 * 6)
        x = self.fc_layers(x)  # (B, 256)
        return x

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


class MaxPoolInstanceMIL(nn.Module):
    def __init__(self, instance_encoder: InstanceEncoder, classifier: Classifier):
        super(MaxPoolInstanceMIL, self).__init__()
        self.instance_encoder = instance_encoder
        self.classifier = classifier

    def forward(self, x: Tuple[Tensor, Tensor]):
        # Case where x is a tuple of (bags, masks) for `mask_collate_fn`
        bags, masks = x       
        B, N = bags.shape[0], bags.shape[1]
        instance_scores_list = []
        
        # Process each instance in the bag (B batches, N instances each)
        for bag, mask in zip(bags, masks):
            instance_features = self.instance_encoder(bag)  # Shape [N, 256]  
            scores = self.classifier(instance_features)  # Shape [N, 1]
            mask_expanded = mask.unsqueeze(1)  # Shape [N, 1]
            masked_scores = scores.masked_fill(mask_expanded == 0, float('-inf'))   
                    
            instance_scores_list.append(masked_scores)
        
        instance_scores = torch.stack(instance_scores_list, dim=0)  # Shape [B, N, 1]        
        bag_scores = torch.max(instance_scores, dim=1)[0]  # Shape [B, 1]
        return bag_scores

