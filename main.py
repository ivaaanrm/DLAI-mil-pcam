
import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from dataset import (
    MILDataset, 
    MILMaskDataLoader,
)

from model import (
    InstanceEncoder,
    Classifier,
    MaxPoolInstanceMIL
)

TRAIN_DIR = "mil-pcam-dummy/train"

def run_inference(mil_model: nn.Module, data_loader: DataLoader):
    """
    Function to run inference for each batch in the provided data loader.

    Args:
        mil_model (nn.Module): The MIL model to perform inference.
        data_loader (DataLoader): The DataLoader that provides batches of data.
    """
    mil_model.eval()

    with torch.no_grad(): 
        for bags, masks, labels in data_loader:
            output = mil_model((bags, masks)) # Forward

            print(f"{output.shape=}")  # Should be [B, 1], B is the batch size
            print(f"{output=}")  

            assert output.shape == (bags.shape[0], 1), f"Output shape mismatch! Expected {[bags.shape[0], 1]}, got {output.shape}"
            assert (output >= 0).all() and (output <= 1).all(), "Output values are out of range!"


def main():
    dataset = MILDataset(root=TRAIN_DIR)
    loader = MILMaskDataLoader(dataset, batch_size=2)
    
    encoder = InstanceEncoder()
    classifier = Classifier(input_size=256, hidden_size=128)
    mil_model = MaxPoolInstanceMIL(encoder, classifier)
    
    run_inference(mil_model, loader)
    


if __name__ == '__main__':
    main()
    

