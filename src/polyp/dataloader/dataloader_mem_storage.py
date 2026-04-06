import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#Random Seeds for repeatable experiments
torch.cuda.manual_seed_all(42) #torch.cuda.manual_seed(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PromptingDataMemorySubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, input_size = 256, augment=False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

        self.transforms_images = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])    # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

        self.transforms_labels = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()])    # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, gt_dir, contrastive_gt = self.dataset[self.indices[idx]]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(gt_dir).convert("L"))

        image = self.transforms_images(image)
        mask = self.transforms_labels(mask)

        return image, mask, contrastive_gt



class PromptingDataMemory(Dataset):

    def __init__(self, img_dirs, gt_dirs, input_size=352, augment=False, device="cuda:0"):
        super(PromptingDataMemory, self).__init__()

        self.img_dirs = img_dirs
        self.gt_dirs = gt_dirs


        self.augment = augment
        self.device = device

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, index):

        image_path = self.img_dirs[index]
        gt_dir = self.gt_dirs[index]
        domain_gt = image_path.split("/")[-4]
        contrastive_gt = int(domain_gt.split("_")[-1])


        return (image_path, gt_dir, contrastive_gt)



