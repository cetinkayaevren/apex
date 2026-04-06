import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


#Random Seeds for repeatable experiments
torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PromptingData(Dataset):

    def __init__(self, img_dirs, gt_dirs, input_size=256, augment=False, device="cuda:4"):
        super(PromptingData, self).__init__()

        self.img_dirs = img_dirs
        self.gt_dirs = gt_dirs

        self.transforms_images = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])            
            ])
        self.transforms_labels = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()]
        )
        
        self.augment = augment
        self.device = device

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, index):

        image_path = self.img_dirs[index]
        mask_dir = self.gt_dirs[index]
        contrastive_gt = 6

        fname = self.img_dirs[index].split("/")[-1]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_dir))

        if len(mask.shape) > 2:  # If mask has multiple channels (e.g., RGB), convert to grayscale
            mask = mask[:, :, 0]
        
        augmented_image = self.transforms_images(image)
        augmented_mask = self.transforms_labels(mask)

        return (augmented_image, augmented_mask, contrastive_gt, fname)