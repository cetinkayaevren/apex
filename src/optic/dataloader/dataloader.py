import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

#Random Seeds for repeatable experiments
torch.cuda.manual_seed_all(42) #torch.cuda.manual_seed(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PromptingData(Dataset):

    def __init__(self, img_dirs, gt_dirs, input_size=352, augment=False, device="cuda:4", image_normalize = True):
        super(PromptingData, self).__init__()

        self.img_dirs = img_dirs
        self.gt_dirs = gt_dirs
        self.input_size = (input_size,input_size)
        self.augment = augment
        self.device = device
        self.img_normalize = image_normalize

    def __len__(self):
        return len(self.img_dirs)
    
    def normalize_image_to_0_1(self, img):
        if len(img.shape) == 4:
            for b in range(img.shape[0]):
                img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
        else:
            img = (img-img.min())/(img.max()-img.min())
        return img

    def __getitem__(self, index):

        image_path = self.img_dirs[index]
        mask_dir = self.gt_dirs[index]
        contrastive_gt = 6      #No need for any method other than APEX, default is 6

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_dir).convert('L')
        
        img = image.resize(self.input_size)
        label = mask.resize(self.input_size, resample=Image.NEAREST)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            img_npy = self.normalize_image_to_0_1(img_npy)
        label_npy = np.array(label)

        mask = np.zeros_like(label_npy)
        mask[label_npy < 255] = 1
        mask[label_npy == 0] = 2
        return img_npy, mask[np.newaxis], contrastive_gt
