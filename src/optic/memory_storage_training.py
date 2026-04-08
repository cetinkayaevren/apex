from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from imutils import paths
from tqdm import tqdm
import torch
from torch.utils.data import random_split
import time
import os
import argparse
from pathlib import Path
import random
import numpy as np
from torchmetrics import Dice
from torchmetrics.segmentation import MeanIoU

import sys
from memory.memory import Memory
from dataloader.dataloader_mem_storage import PromptingDataMemory, PromptingDataMemorySubset
from dataloader.dataloader import PromptingData

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from loss.loss_optic.multi_ce_dice_loss import DiceCELoss
from load_config_swin import swin_config
import logging

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.UNet.UNet import UNet
from models.TransUNet.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from models.TransUNet.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.SwinUnet.networks.vision_transformer import SwinUnet as Swin_ViT_seg
from models.ResUNet.ResUnet import ResUnet

#Random Seeds for repeatable experiments
torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator1 = torch.Generator().manual_seed(42)

#Select Model
models_available = ["UNet", "ResUnet", "TransUNet", "SwinUNet"]

def parse_model_selection():
    """
    Model selection priority:
      1) --model <name>  (CLI argument)
      2) APEX_MODEL env var
      3) default: UNet
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=str, default=None, choices=models_available,
                        help="Model name: UNet, ResUnet, TransUNet, SwinUNet")
    args, _ = parser.parse_known_args()

    if args.model is not None:
        return args.model

    env_model = os.getenv("APEX_MODEL")
    if env_model:
        if env_model not in models_available:
            raise ValueError(f"Invalid APEX_MODEL='{env_model}'. Choose from {models_available}.")
        return env_model

    return "UNet"

selected_model = parse_model_selection()

#Default Prompt Parameters
PROMPT_DIM = 6
MEM_SIZE = 150
TEMPERATURE = 1
CONT_TEMP = 0.07 
ADDRESSING_TYPE = "soft"

#Default Training Parameters
N_CLASS = 3
LEARNING_RATE = 1e-3 
EPOCHS = 150
WARMUP_EPOCH = 0
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-2
BETAS = (0.9,0.99)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if selected_model == "SwinUNet":
    INPUT_SIZE = 224
else:
    INPUT_SIZE = 352

#TransUNet Parameters
VIT_NAME = "R50-ViT-B_16"
PATCHES_SIZE = 16
config_vit = CONFIGS_ViT_seg[VIT_NAME]
config_vit.n_classes = 3
config_vit.n_skip = 3
config_vit.patches.size = (PATCHES_SIZE, PATCHES_SIZE)
config_vit.patches.grid = (int(INPUT_SIZE/PATCHES_SIZE), int(INPUT_SIZE/PATCHES_SIZE))

#SwinUNet Parameters
config_swin = swin_config()

# Models
models = {
    "UNet": UNet(img_ch=3, output_ch=3),
    "ResUnet": ResUnet(num_classes=3),
    "TransUNet": ViT_seg(config_vit, img_size=INPUT_SIZE, num_classes=N_CLASS),
    "SwinUNet": Swin_ViT_seg(config_swin, img_size=INPUT_SIZE, num_classes=N_CLASS),
}

#Datasets for training prompt
source_domain_names = ["REFUGE", "REFUGE_Valid"]
target_domain_names = ["REFUGE", "REFUGE_Valid", "Drishti_GS", "RIM_ONE_r3"]


print(f"{selected_model} is selected.")

#Create Logging Config
logging.basicConfig(level=logging.INFO, filename=f"optic_memory_training_{selected_model}_mem_slot_{MEM_SIZE}.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Training Data Parameters
print(f"{source_domain_names} are starting to prepare for memory training")

# Portable paths (optionally overridable by environment variables)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("APEX_PROJECT_ROOT", SCRIPT_DIR.parents[1]))
DATA_ROOT = Path(os.getenv("APEX_DATA_ROOT", PROJECT_ROOT))
DATASET_ROOT = DATA_ROOT / "dataset_exp" / "Fundus"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optic"

TRAIN_IMAGE_DOMAIN_1 = DATASET_ROOT / f"{source_domain_names[0]}_0" / "train" / "image"
TRAIN_MASK_DOMAIN_1 = DATASET_ROOT / f"{source_domain_names[0]}_0" / "train" / "mask"

TRAIN_IMAGE_DOMAIN_2 = DATASET_ROOT / f"{source_domain_names[0]}_augmented_domain_1" / "train" / "image"
TRAIN_MASK_DOMAIN_2 = DATASET_ROOT / f"{source_domain_names[0]}_augmented_domain_1" / "train" / "mask"

TRAIN_IMAGE_DOMAIN_3 = DATASET_ROOT / f"{source_domain_names[0]}_augmented_domain_2" / "train" / "image"
TRAIN_MASK_DOMAIN_3 = DATASET_ROOT / f"{source_domain_names[0]}_augmented_domain_2" / "train" / "mask"

TRAIN_IMAGE_DOMAIN_4 = DATASET_ROOT / f"{source_domain_names[1]}_3" / "train" / "image"
TRAIN_MASK_DOMAIN_4 = DATASET_ROOT / f"{source_domain_names[1]}_3" / "train" / "mask"

TRAIN_IMAGE_DOMAIN_5 = DATASET_ROOT / f"{source_domain_names[1]}_augmented_domain_4" / "train" / "image"
TRAIN_MASK_DOMAIN_5 = DATASET_ROOT / f"{source_domain_names[1]}_augmented_domain_4" / "train" / "mask"

TRAIN_IMAGE_DOMAIN_6 = DATASET_ROOT / f"{source_domain_names[1]}_augmented_domain_5" / "train" / "image"
TRAIN_MASK_DOMAIN_6 = DATASET_ROOT / f"{source_domain_names[1]}_augmented_domain_5" / "train" / "mask"

TARGET_IMAGE_DOMAIN = DATASET_ROOT / f"{target_domain_names[0]}_0" / "test" / "image"
TARGET_MASK_DOMAIN = DATASET_ROOT / f"{target_domain_names[0]}_0" / "test" / "mask"

TARGET_IMAGE_DOMAIN_2 = DATASET_ROOT / f"{target_domain_names[1]}_3" / "test" / "image"
TARGET_MASK_DOMAIN_2 = DATASET_ROOT / f"{target_domain_names[1]}_3" / "test" / "mask"

TARGET_IMAGE_DOMAIN_3 = DATASET_ROOT / f"{target_domain_names[2]}_0" / "test" / "image"
TARGET_MASK_DOMAIN_3 = DATASET_ROOT / f"{target_domain_names[2]}_0" / "test" / "mask"

TARGET_IMAGE_DOMAIN_4 = DATASET_ROOT / target_domain_names[3] / "test" / "image"
TARGET_MASK_DOMAIN_4 = DATASET_ROOT / target_domain_names[3] / "test" / "mask"

MEMORY_WEIGHTS_DIR = OUTPUT_DIR / "memory_weights"
MEMORY_BEST_PATH = MEMORY_WEIGHTS_DIR / f"prompt_memory_optic_{selected_model}_prompt_{PROMPT_DIM}_epoch_{EPOCHS}_lr_{LEARNING_RATE}_add_{ADDRESSING_TYPE}_mem_{MEM_SIZE}_temp_{TEMPERATURE}_best.pth"
MEMORY_LAST_PATH = MEMORY_WEIGHTS_DIR / f"prompt_memory_optic_{selected_model}_prompt_{PROMPT_DIM}_epoch_{EPOCHS}_lr_{LEARNING_RATE}_add_{ADDRESSING_TYPE}_mem_{MEM_SIZE}_temp_{TEMPERATURE}_last.pth"
PLOT_PATH = OUTPUT_DIR / f"plot_memory_{selected_model}_prompt_{PROMPT_DIM}_epoch_{EPOCHS}_lr_{LEARNING_RATE}_add_{ADDRESSING_TYPE}_mem_{MEM_SIZE}_temp_{TEMPERATURE}.png"

PRETRAINED_WEIGHTS_DIR = OUTPUT_DIR / "pretrained_weights"
PRETRAIN_PATHS = {
    "UNet":      PRETRAINED_WEIGHTS_DIR / "pretrain-UNet.pth",
    "ResUnet":   PRETRAINED_WEIGHTS_DIR / "pretrain-ResUnet.pth",
    "TransUNet": PRETRAINED_WEIGHTS_DIR / "pretrain-TransUNet.pth",
    "SwinUNet":  PRETRAINED_WEIGHTS_DIR / "pretrain-SwinUNet.pth",
}
PRETRAIN_PATH = PRETRAIN_PATHS[selected_model]

# Take Data
train_image_path_domain_1 = sorted(list(paths.list_images(str(TRAIN_IMAGE_DOMAIN_1))))
train_mask_path_domain_1 = sorted(list(paths.list_images(str(TRAIN_MASK_DOMAIN_1))))

train_image_path_domain_2 = sorted(list(paths.list_images(str(TRAIN_IMAGE_DOMAIN_2))))
train_mask_path_domain_2 = sorted(list(paths.list_images(str(TRAIN_MASK_DOMAIN_2))))

train_image_path_domain_3 = sorted(list(paths.list_images(str(TRAIN_IMAGE_DOMAIN_3))))
train_mask_path_domain_3 = sorted(list(paths.list_images(str(TRAIN_MASK_DOMAIN_3))))

train_image_path_domain_4 = sorted(list(paths.list_images(str(TRAIN_IMAGE_DOMAIN_4))))
train_mask_path_domain_4 = sorted(list(paths.list_images(str(TRAIN_MASK_DOMAIN_4))))

train_image_path_domain_5 = sorted(list(paths.list_images(str(TRAIN_IMAGE_DOMAIN_5))))
train_mask_path_domain_5 = sorted(list(paths.list_images(str(TRAIN_MASK_DOMAIN_5))))

train_image_path_domain_6 = sorted(list(paths.list_images(str(TRAIN_IMAGE_DOMAIN_6))))
train_mask_path_domain_6 = sorted(list(paths.list_images(str(TRAIN_MASK_DOMAIN_6))))

target_image_path_domain = sorted(list(paths.list_images(str(TARGET_IMAGE_DOMAIN))))
target_mask_path_domain = sorted(list(paths.list_images(str(TARGET_MASK_DOMAIN))))

target_image_path_domain_2 = sorted(list(paths.list_images(str(TARGET_IMAGE_DOMAIN_2))))
target_mask_path_domain_2 = sorted(list(paths.list_images(str(TARGET_MASK_DOMAIN_2))))

target_image_path_domain_3 = sorted(list(paths.list_images(str(TARGET_IMAGE_DOMAIN_3))))
target_mask_path_domain_3 = sorted(list(paths.list_images(str(TARGET_MASK_DOMAIN_3))))

target_image_path_domain_4 = sorted(list(paths.list_images(str(TARGET_IMAGE_DOMAIN_4))))
target_mask_path_domain_4 = sorted(list(paths.list_images(str(TARGET_MASK_DOMAIN_4))))

total_train_image_path = sorted(train_image_path_domain_1 + 
                                train_image_path_domain_2 +
                                train_image_path_domain_3 +
                                train_image_path_domain_4 +
                                train_image_path_domain_5 +
                                train_image_path_domain_6)

total_train_mask_path = sorted(train_mask_path_domain_1 + 
                                train_mask_path_domain_2 +
                                train_mask_path_domain_3 +
                               train_mask_path_domain_4 + 
                               train_mask_path_domain_5 +
                               train_mask_path_domain_6)


# For Training
dataset_image_prompt = PromptingDataMemory(img_dirs = total_train_image_path, 
                                            gt_dirs = total_train_mask_path, 
                                            input_size=INPUT_SIZE, 
                                            augment = False, 
                                            device=DEVICE)

target_image_prompt = PromptingData(img_dirs = target_image_path_domain, 
                                            gt_dirs = target_mask_path_domain, 
                                            input_size=INPUT_SIZE, 
                                            augment = False, 
                                            device=DEVICE)

target_image_prompt_2 = PromptingData(img_dirs = target_image_path_domain_2, 
                                            gt_dirs = target_mask_path_domain_2, 
                                            input_size=INPUT_SIZE, 
                                            augment = False, 
                                            device=DEVICE)

target_image_prompt_3 = PromptingData(img_dirs = target_image_path_domain_3, 
                                            gt_dirs = target_mask_path_domain_3, 
                                            input_size=INPUT_SIZE, 
                                            augment = False, 
                                            device=DEVICE)

target_image_prompt_4 = PromptingData(img_dirs = target_image_path_domain_4, 
                                            gt_dirs = target_mask_path_domain_4, 
                                            input_size=INPUT_SIZE, 
                                            augment = False, 
                                            device=DEVICE)

train_indices, val_indices= random_split(range(len(dataset_image_prompt)), [0.9, 0.1], generator=generator1)

train_dataset = PromptingDataMemorySubset(dataset_image_prompt, train_indices, input_size=INPUT_SIZE, augment=False, img_normalize = True)
val_dataset = PromptingDataMemorySubset(dataset_image_prompt, val_indices, input_size=INPUT_SIZE, augment=False, img_normalize = True)

def worker_init_fn(worker_id):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

trainLoader = DataLoader(train_dataset, shuffle=True,
                            batch_size=BATCH_SIZE, 
                            pin_memory=True,
                            num_workers=8,
                            worker_init_fn=worker_init_fn,
                            generator=generator1
                            )

valLoader = DataLoader(val_dataset, shuffle=False,
                            batch_size=BATCH_SIZE, #1
                            pin_memory=True,
                            num_workers=1,
                            worker_init_fn=worker_init_fn,
                            generator=generator1
                            )

testLoader = DataLoader(target_image_prompt, shuffle=False,
                            batch_size=1, 
                            pin_memory=True,
                            num_workers=1,
                            worker_init_fn=worker_init_fn,
                            generator=generator1
                            )

testLoader2 = DataLoader(target_image_prompt_2, shuffle=False,
                            batch_size=1, 
                            pin_memory=True,
                            num_workers=1,
                            worker_init_fn=worker_init_fn,
                            generator=generator1
                            )

testLoader3 = DataLoader(target_image_prompt_3, shuffle=False,
                            batch_size=1, 
                            pin_memory=True,
                            num_workers=1,
                            worker_init_fn=worker_init_fn,
                            generator=generator1
                            )

testLoader4 = DataLoader(target_image_prompt_4, shuffle=False,
                            batch_size=1, 
                            pin_memory=True,
                            num_workers=1,
                            worker_init_fn=worker_init_fn,
                            generator=generator1
                            )

#Model Loading
model = models[selected_model].to(DEVICE)
model.load_state_dict(torch.load(PRETRAIN_PATH, weights_only=True, map_location=DEVICE))

# Freeze segmentation model parameters
for param in model.parameters():
    param.requires_grad = False

#Initialize Memory
prompt_memory = Memory(memory_size=MEM_SIZE, prompt_size=PROMPT_DIM, temperature=TEMPERATURE, cont_temp=CONT_TEMP, device=DEVICE).to(DEVICE)

#Loss loading
seg_loss = DiceCELoss(dice_weight=0.5, ce_weight=0.5, num_class=3)

#Set Optimizer to learn prompt
optimizer = AdamW(prompt_memory.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#Set scheduler
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

trainSteps = len(train_dataset) // BATCH_SIZE
valSteps = len(val_dataset) // BATCH_SIZE
testSteps = len(target_image_prompt) // 1
testSteps2 = len(target_image_prompt_2) // 1
testSteps3 = len(target_image_prompt_3) // 1
testSteps4 = len(target_image_prompt_4) // 1

# Training
H = {"train_loss": [], "val_loss": []}
best_loss = 1e8
best_validation_loss = 1e8
best_val_dice=0.
best_avg_dice=0.

print("[INFO] Output folders are creating")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

dice = Dice(average="macro", num_classes=3, ignore_index=None).to(DEVICE)
miou = MeanIoU(num_classes=3, include_background=True, input_format='index').to(DEVICE)

print("[INFO] training the memory...")
logging.info("[INFO] training the memory...")
startTime = time.time()


for e in tqdm(range(EPOCHS)):
    # set the model in evaluation mode (no training)
    model.eval()
    #set the prompt memory in training mode (update prompt memory)
    prompt_memory.train()
    
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    totalContLoss = 0
    totalConValLoss = 0
    for (i, (images, masks, domain_gt)) in enumerate(trainLoader):
        # send the input to the device
        (images, masks, domain_gt) = (images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_gt.to(DEVICE))

        # perform a forward pass and calculate the training loss
        prompted_images, addressing_vec, pred_prompt, contrastive_loss = prompt_memory(images, domain_gt)
        if selected_model == "ResUnet":		#Res-UNet
            pred = model(prompted_images)[0]
        else:
            pred = model(prompted_images)
        
        _loss = seg_loss(pred, masks) + 0.1*contrastive_loss

        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        # add the loss to the total training loss so far
        totalTrainLoss += _loss
        totalContLoss += contrastive_loss

    # Step the scheduler
    scheduler.step()

    # Print learning rate for visualization
    current_lr = optimizer.param_groups[0]['lr']

    total_val_dice = 0
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        prompt_memory.eval()
        # loop over the val set
        for (images, masks, domain_gt) in valLoader:
            # send the input to the device
            (images, masks, domain_gt) = (images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_gt.to(DEVICE))
            #Concatenate the prompt to image

            prompted_images, addressing_vec, pred_prompt, contrastive_loss = prompt_memory(images, domain_gt)

            if selected_model == "ResUnet":		#Res-UNet
                pred = model(prompted_images)[0]
            else:
                pred = model(prompted_images)

            pred_class = pred.argmax(dim=1)

            _loss_val = seg_loss(pred, masks) + 0.1*contrastive_loss

            total_val_dice += dice(pred_class, masks.long().to(DEVICE)) / valSteps
            totalValLoss += _loss_val
            totalConValLoss += contrastive_loss

#----------------- Testing Phase ---------------------
    total_dice = 0.
    total_miou = 0.
    with torch.no_grad():
        model.eval()
        prompt_memory.eval()

        for (i, (images, masks, domain_gt)) in enumerate(testLoader):
            images, labels, domain_gt = images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_gt.to(DEVICE)
            
            prompted_images, addressing_vec, pred_prompt, _ = prompt_memory(images, domain_gt)

            if selected_model == "ResUnet":		#Res-UNet
                pred = model(prompted_images)[0]
            else:
                pred = model(prompted_images)

            pred_class = pred.argmax(dim=1)      

            # Accumulate metrics
            total_dice += dice(pred_class, labels.long().to(DEVICE)) / testSteps
            total_miou += miou(pred_class, labels.long().to(DEVICE)) / testSteps

#----------------- Testing Phase 2---------------------
    total_dice2 = 0.
    total_miou2 = 0.
    with torch.no_grad():
        model.eval()
        prompt_memory.eval()

        for (i, (images, masks, domain_gt)) in enumerate(testLoader2):
            images, labels, domain_gt = images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_gt.to(DEVICE)
            
            prompted_images, addressing_vec, pred_prompt, _ = prompt_memory(images, domain_gt)

            if selected_model == "ResUnet":		#Res-UNet
                pred = model(prompted_images)[0]
            else:
                pred = model(prompted_images)

            pred_class = pred.argmax(dim=1)          

            # Accumulate metrics
            total_dice2 += dice(pred_class, labels.long().to(DEVICE)) / testSteps2
            total_miou2 += miou(pred_class, labels.long().to(DEVICE)) / testSteps2

#----------------- Testing Phase 3---------------------
    total_dice3 = 0.
    total_miou3 = 0.
    with torch.no_grad():
        model.eval()
        prompt_memory.eval()

        for (i, (images, masks, domain_gt)) in enumerate(testLoader3):
            images, labels, domain_gt = images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_gt.to(DEVICE)
            
            prompted_images, addressing_vec, pred_prompt, _ = prompt_memory(images, domain_gt)

            if selected_model == "ResUnet":		#Res-UNet
                pred = model(prompted_images)[0]
            else:
                pred = model(prompted_images)

            pred_class = pred.argmax(dim=1)             

            # Accumulate metrics
            total_dice3 += dice(pred_class, labels.long().to(DEVICE)) / testSteps3
            total_miou3 += miou(pred_class, labels.long().to(DEVICE)) / testSteps3

#----------------- Testing Phase 4---------------------
    total_dice4 = 0.
    total_miou4 = 0.

    with torch.no_grad():
        model.eval()
        prompt_memory.eval()

        for (i, (images, masks, domain_gt)) in enumerate(testLoader4):
            images, labels, domain_gt = images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_gt.to(DEVICE)
            
            prompted_images, addressing_vec, pred_prompt, _ = prompt_memory(images, domain_gt)

            if selected_model == "ResUnet":		#Res-UNet
                pred = model(prompted_images)[0]
            else:
                pred = model(prompted_images)

            pred_class = pred.argmax(dim=1)           

            # Accumulate metrics
            total_dice4 += dice(pred_class, labels.long().to(DEVICE)) / testSteps4
            total_miou4 += miou(pred_class, labels.long().to(DEVICE)) / testSteps4

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgContLoss = totalContLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    avgContValLoss = totalConValLoss / valSteps

    average_total_dice = (total_dice.cpu().detach().numpy() + total_dice2.cpu().detach().numpy() + total_dice3.cpu().detach().numpy() + total_dice4.cpu().detach().numpy())/4

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print(f"""Train loss: {avgTrainLoss}, 
          Train Cont. Loss: {avgContLoss}, 
          Train Cont. Val Loss: {avgContValLoss},
          Validation DICE: {total_val_dice.cpu().detach().numpy()},
          Avg. Total DICE: {average_total_dice},
          {target_domain_names[0]} DICE:{total_dice.cpu().detach().numpy()}, 
          {target_domain_names[0]} mIoU:{total_miou.cpu().detach().numpy()}, 
          {target_domain_names[1]} DICE:{total_dice2.cpu().detach().numpy()}, 
          {target_domain_names[1]} mIoU:{total_miou2.cpu().detach().numpy()}, 
          {target_domain_names[2]} DICE:{total_dice3.cpu().detach().numpy()}, 
          {target_domain_names[2]} mIoU:{total_miou3.cpu().detach().numpy()}, 
          {target_domain_names[3]} DICE:{total_dice4.cpu().detach().numpy()}, 
          {target_domain_names[3]} mIoU:{total_miou4.cpu().detach().numpy()}, 
          lr:{current_lr}""")
    
    logging.info(f"[INFO] EPOCH: {e + 1}/{EPOCHS}")
    logging.info(f"Train loss: {avgTrainLoss}, Cont Loss: {avgContLoss}, Cont Val Loss: {avgContValLoss}, Validation loss: {avgValLoss}, Avg. Total DICE: {average_total_dice}, {target_domain_names[0]} DICE: {total_dice.cpu().detach().numpy()}, {target_domain_names[0]} mIoU: {total_miou.cpu().detach().numpy(),}, {target_domain_names[1]} DICE: {total_dice2.cpu().detach().numpy()}, {target_domain_names[1]} mIoU: {total_miou2.cpu().detach().numpy()}, {target_domain_names[2]} DICE: {total_dice3.cpu().detach().numpy()}, {target_domain_names[2]} mIoU: {total_miou3.cpu().detach().numpy()}, {target_domain_names[3]} DICE: {total_dice4.cpu().detach().numpy()}, {target_domain_names[3]} mIoU: {total_miou4.cpu().detach().numpy()}")
    
    if average_total_dice>best_avg_dice:
        print("Best memory is written")
        logging.info("Best memory is written")
        best_avg_dice=average_total_dice
        torch.save(prompt_memory.state_dict(), MEMORY_BEST_PATH)
    torch.save(prompt_memory.state_dict(), MEMORY_LAST_PATH)

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the prompt: {:.2f}s".format(
    endTime - startTime))

print("[INFO] DONE")