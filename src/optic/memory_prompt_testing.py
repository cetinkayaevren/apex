from torch.utils.data import DataLoader
from imutils import paths
import torch
import os
import argparse
from pathlib import Path
import random
import numpy as np
from torchmetrics import Dice, Precision, Recall
from torchmetrics.segmentation import MeanIoU
import sys
from load_config_swin import swin_config

from memory.memory import Memory
from dataloader.dataloader import PromptingData

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.UNet.UNet import UNet
from models.SwinUnet.networks.vision_transformer import SwinUnet as Swin_ViT_seg
from models.TransUNet.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from models.TransUNet.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.ResUNet.ResUnet import ResUnet


#Random Seeds for repeatable experiments
torch.cuda.manual_seed_all(42) #torch.cuda.manual_seed(42)
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

#Datasets for training prompt
target_domain_names = ["REFUGE_0", "REFUGE_Valid_3", "Drishti_GS_0",  "RIM_ONE_r3"]
trained_prompt_domain_name = "ORIGA"
lambda_factor = 0.4
print(f"{selected_model} is selected.")

#Default Prompt Parameters
PROMPT_DIM = 6
MEM_SIZE = 150
TEMPERATURE= 1
ADDRESSING_TYPE = "soft"

#Default Training Parameters
if selected_model == "SwinUNet":
    INPUT_SIZE = 224
else:
    INPUT_SIZE = 352

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCHS= 150
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CLASS = 3

# Portable paths (optionally overridable by environment variables)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("APEX_PROJECT_ROOT", SCRIPT_DIR.parents[1]))
DATA_ROOT = Path(os.getenv("APEX_DATA_ROOT", PROJECT_ROOT))
DATASET_ROOT = DATA_ROOT / "dataset_exp" / "Fundus"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optic"

def resolve_memory_path(model_name: str) -> Path:
    """
    Resolve prompt memory weight path from the outputs folder.
    Priority:
      1) APEX_MEMORY_PATH env var (explicit file)
      2) Any .pth file in memory_weights containing the model name
         (prefer files containing 'best' or 'last')
    """
    memory_override = os.getenv("APEX_MEMORY_PATH")
    if memory_override:
        override_path = Path(memory_override)
        if override_path.exists():
            return override_path
        raise FileNotFoundError(f"APEX_MEMORY_PATH does not exist: {override_path}")

    memory_weights_dir = OUTPUT_DIR / "memory_weights"
    if not memory_weights_dir.exists():
        raise FileNotFoundError(f"Memory weights folder not found: {memory_weights_dir}")

    candidates = sorted(memory_weights_dir.glob(f"*{model_name}*.pth"))
    if not candidates:
        candidates = sorted(memory_weights_dir.glob("*.pth"))

    if not candidates:
        raise FileNotFoundError(
            f"No prompt memory weights found in {memory_weights_dir}. "
            f"Please place .pth files there (preferably containing '{model_name}' in filename)."
        )

    preferred = [p for p in candidates if ("best" in p.name.lower() or "last" in p.name.lower())]
    return preferred[0] if preferred else candidates[0]

MEMORY_PATH = resolve_memory_path(selected_model)

# TransUNet parameters
VIT_NAME = "R50-ViT-B_16"
PATCHES_SIZE = 16
config_vit = CONFIGS_ViT_seg[VIT_NAME]
config_vit.n_classes = 3
config_vit.n_skip = 3
config_vit.patches.size = (PATCHES_SIZE, PATCHES_SIZE)
config_vit.patches.grid = (int(INPUT_SIZE/PATCHES_SIZE), int(INPUT_SIZE/PATCHES_SIZE))

#SwinUNet Parameters
config_swin = swin_config()

def worker_init_fn(worker_id):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

# Models
models = {
    "UNet": UNet(img_ch=3, output_ch=3),
    "ResUnet": ResUnet(num_classes=3),
    "TransUNet": ViT_seg(config_vit, img_size=INPUT_SIZE, num_classes=N_CLASS),
    "SwinUNet": Swin_ViT_seg(config_swin, img_size=INPUT_SIZE, num_classes=N_CLASS),
}

#Performance Metrics
dice = Dice(average="macro", num_classes=3, ignore_index=None).to(DEVICE)
miou = MeanIoU(num_classes=3, include_background=True, input_format='index').to(DEVICE)
precision = Precision(task="multiclass", average='macro', num_classes=3).to(DEVICE)
recall = Recall(task="multiclass", average='macro', num_classes=3).to(DEVICE)

main_dice_list = []
main_iou_list = []

for target_domain_name in target_domain_names:
    print(f"{target_domain_name} is starting to prepare for testing the memory")

    TEST_IMAGE_PATH = DATASET_ROOT / target_domain_name / "test" / "image"
    TEST_MASK_PATH = DATASET_ROOT / target_domain_name / "test" / "mask"

    # Take Data
    test_image_path = sorted(list(paths.list_images(str(TEST_IMAGE_PATH))))
    test_mask_path = sorted(list(paths.list_images(str(TEST_MASK_PATH))))

    test_dataset_masks = PromptingData(img_dirs = test_image_path,
                                       gt_dirs = test_mask_path, 
                                       input_size=INPUT_SIZE, 
                                       augment = False,
                                       device = DEVICE)

    testLoader = DataLoader(test_dataset_masks, 
                            shuffle=False,
                            batch_size=BATCH_SIZE, 
                            pin_memory=True,
                            num_workers=1,
                            worker_init_fn=worker_init_fn,
                            generator=generator1)

    testSteps = len(test_dataset_masks) // BATCH_SIZE

    #Model Loading
    PRETRAINED_WEIGHTS_DIR = OUTPUT_DIR / "pretrained_weights"
    PRETRAIN_PATHS = {
        "UNet":      PRETRAINED_WEIGHTS_DIR / "pretrain-UNet.pth",
        "ResUnet":   PRETRAINED_WEIGHTS_DIR / "pretrain-ResUnet.pth",
        "TransUNet": PRETRAINED_WEIGHTS_DIR / "pretrain-TransUNet.pth",
        "SwinUNet":  PRETRAINED_WEIGHTS_DIR / "pretrain-SwinUNet.pth",
    }
    PRETRAIN_PATH = PRETRAIN_PATHS[selected_model]

    model = models[selected_model].to(DEVICE)
    model.load_state_dict(torch.load(PRETRAIN_PATH, weights_only=True, map_location=DEVICE))

    #Memory bank loading
    prompt_memory = Memory(memory_size = MEM_SIZE, prompt_size=PROMPT_DIM, temperature=TEMPERATURE, device=DEVICE, lambda_factor=lambda_factor).to(DEVICE)
    prompt_memory.load_state_dict(torch.load(MEMORY_PATH, weights_only=True), strict=True)

    # Freeze segmentation model parameters
    for param in model.parameters():
        param.requires_grad = False

    #----------------- Testing Phase ---------------------
    total_test_dice = 0

    total_dice = 0.
    total_miou = 0.
    total_precision = 0.
    total_recall = 0.
    dice_scores = []
    miou_scores = []
    with torch.no_grad():
        model.eval()
        prompt_memory.eval()

        for (i, (images, masks, domain_labels)) in enumerate(testLoader):

            images, labels, domain_labels = images.to(DEVICE), masks.squeeze(1).to(DEVICE).long(), domain_labels.to(DEVICE)
            prompted_images, addressing_vector, _, _ = prompt_memory(images, domain_labels)

            if selected_model == "ResUnet":
                pred = model(prompted_images)[0]
            else:   # Other models
                pred = model(prompted_images) 

            pred_class = pred.argmax(dim=1)

            # Accumulate metrics
            dice_score = dice(pred_class, labels.to(DEVICE))
            dice_scores.append(dice_score.cpu().detach().numpy())
            miou_score = miou(pred_class, labels.to(DEVICE))
            miou_scores.append(miou_score.cpu().detach().numpy())

            total_dice += dice_score / testSteps
            total_miou += miou_score / testSteps
            total_precision += precision(pred_class, labels.long().to(DEVICE)) / testSteps
            total_recall += recall(pred_class, labels.long().to(DEVICE)) / testSteps

    main_dice_list.append(dice_scores)
    main_iou_list.append(miou_scores)

    # Convert to numpy
    print("DICE: ", total_dice.cpu().detach().numpy())
    print("mIoU: ", total_miou.cpu().detach().numpy())
    print("Precision: ", total_precision.cpu().detach().numpy())
    print("Recall: ", total_recall.cpu().detach().numpy())
    print("Std DICE: ", np.std(dice_scores))
    print("Std mIoU: ", np.std(miou_scores))

main_dice_list_x = sum(main_dice_list, [])
main_iou_list_x = sum(main_iou_list, [])

print("Main Std DICE: ", np.std(main_dice_list_x))
print("Main Std mIoU: ", np.std(main_iou_list_x))
