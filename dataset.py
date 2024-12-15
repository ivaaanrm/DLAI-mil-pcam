# Ivan Romero Moreno, DLAI - ETSETB

import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch import Tensor
from pathlib import Path
from typing import Callable, List, Tuple
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


class MILDataset(Dataset):
    def __init__(self, root : str, transform: Callable=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.labels =  pd.read_csv(self.root / "labels.csv")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx : int) -> Tuple[Tensor, Tensor]:
        bag_id = self.root / 'bags' / self.labels.iloc[idx, 0]
        label = torch.tensor(self.labels.iloc[idx, 1])

        image_tensors = []
        for img_path in list(bag_id.glob("*.png")):  
            image = read_image(str(img_path))
            if self.transform:
                image = self.transform(image)
            image_tensors.append(image)
          
        bag = torch.stack(image_tensors) if image_tensors else torch.empty(0)

        return bag, label

    def show_bag(self, idx: int, nrow: int = 5) -> None:
        """
        Displays the patches of a bag in a grid.
        """
        bag, label = self[idx]  # bag is a Tensor of shape [N, 3, 96, 96]

        if bag.size(0) == 0:
            print(f"No patches available in bag {idx}.")
            return

        # Determine grid dimensions
        num_images = bag.size(0)
        ncol = (num_images + nrow - 1) // nrow  # number of columns

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < num_images:
                patch = bag[i].permute(1, 2, 0).cpu().numpy() 
                ax.imshow(patch)
                ax.axis("off")
            else:
                ax.axis("off")

        plt.suptitle(f"Bag {idx} - Label: {label.item()}", fontsize=16)
        plt.tight_layout()
        plt.show()


class MILMaskDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=mask_mil_collate_fn)

def ls_mil_collate_fn(batch : List[Tuple[Tensor, Tensor]]) -> Tuple[List[Tensor], Tensor]:
    bags = [bag for bag, _ in batch] 
    labels = torch.stack([label for _, label in batch]) 
    return bags, labels

def mask_mil_collate_fn(batch : List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    # get the bags and labels
    bags, labels = zip(*batch)
    # get the maximum number of instances in the same bag (for this batch)
    max_inst = max([b.size(0) for b in bags])
    # pad each bag with 0's, and create binary masks
    pad_bags = [torch.cat([b, torch.zeros(max_inst - b.size(0), *b.shape[1:])], dim=0) for b in bags]
    masks = [torch.cat([torch.ones(b.size(0)), torch.zeros(max_inst - b.size(0))], dim=0).bool() for b in bags]
    # stack the bags and masks
    return torch.stack(pad_bags), torch.stack(masks), torch.stack(labels)




if __name__ == "__main__":
    TRAIN_DIR = "mil-pcam-dummy/train"
    
    _tmp_dataset = MILDataset(root=TRAIN_DIR)
    # length of the dataset
    print(f"Length of the dataset: {len(_tmp_dataset)}")
    # get a sample
    bag, y = _tmp_dataset[1]
    print(f"Bag shape: {bag.shape}")
    print(f"Label: {y}")
    
    # verify the collate_fn works
    _tmp_loader = DataLoader(
        _tmp_dataset, 
        batch_size=2,
        collate_fn=ls_mil_collate_fn
    )
    bags, labels = next(iter(_tmp_loader))
    print(type(bags))
    print(f"Number of bags: {len(bags)}")
    print(f"Number of labels: {labels.size(0)}")
    print(f"Number of instances in the first bag: {bags[0].size(0)}")
    
    _tmp_dataset.show_bag(0)
