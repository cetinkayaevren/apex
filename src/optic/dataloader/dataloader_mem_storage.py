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

class PromptingDataMemorySubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, input_size = 352, augment=False, img_normalize = True):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment
        self.input_size = (input_size,input_size)
        self.img_normalize = img_normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, gt_dir, contrastive_gt = self.dataset[self.indices[idx]]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(gt_dir).convert("L")

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

    def normalize_image_to_0_1(self, img):
        if len(img.shape) == 4:
            for b in range(img.shape[0]):
                img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
        else:
            img = (img-img.min())/(img.max()-img.min())
        return img

class PromptingDataMemory(Dataset):

    def __init__(self, img_dirs, gt_dirs, input_size=352, augment=False, device="cuda:0"):
        super(PromptingDataMemory, self).__init__()

        self.img_dirs = img_dirs
        #self.prompt_dirs = prompt_dirs
        self.gt_dirs = gt_dirs

        #self.transform = ImageMaskTransform(self.transforms_images, self.transforms_labels)

        self.augment = augment
        self.device = device

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, index):

        image_path = self.img_dirs[index]
        gt_dir = self.gt_dirs[index]
        domain_gt = image_path.split("/")[-4]
        contrastive_gt = int(domain_gt.split("_")[-1])

        if contrastive_gt==1 or contrastive_gt==2:
            contrastive_gt = 0
        if contrastive_gt==4 or contrastive_gt==3 or contrastive_gt==5:
            contrastive_gt = 1
        
        return (image_path, gt_dir, contrastive_gt)



