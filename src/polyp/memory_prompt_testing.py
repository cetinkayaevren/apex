from dataloader.dataloader import PromptingData
from prompt.vis_prompt import VisualPrompt
from torch.utils.data import DataLoader
from torchvision import transforms
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
from configs.load_config_swin import swin_config
from memory.memory import Memory as MemoryCNN   # for DUCK-Net and PraNet
from memory.memory_trans_swin import Memory as MemoryTransformer  # for TransUNet SwinUNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.SwinUnet.networks.vision_transformer import SwinUnet as Swin_ViT_seg
from models.TransUNet.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from models.TransUNet.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.PraNet.PraNet_Res2Net import PraNet

#Random Seeds for repeatable experiments
torch.cuda.manual_seed_all(42) #torch.cuda.manual_seed(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator1 = torch.Generator().manual_seed(42) # For train-val split

#Select Model
models_available = ["PraNet", "TransUNet", "SwinUNet"]

def parse_model_selection():
    """
    Model selection priority:
      1) --model <name>  (CLI argument)
      2) APEX_MODEL env var
      3) default: PraNet
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=str, default=None, choices=models_available,
                        help="Model name: PraNet, TransUNet, SwinUNet")
    args, _ = parser.parse_known_args()

    if args.model is not None:
        return args.model

    env_model = os.getenv("APEX_MODEL")
    if env_model:
        if env_model not in models_available:
            raise ValueError(f"Invalid APEX_MODEL='{env_model}'. Choose from {models_available}.")
        return env_model

    return "PraNet"

selected_model = parse_model_selection()

#Datasets for training prompt
target_domain_names = ["CVC-ClinicDB_0", "Kvasir-SEG_3", "ETIS-LaribPolypDB", "CVC-ColonDB"]
trained_prompt_domain_name = "BKAI"
print(f"{selected_model} is selected.")

#Default Prompt Parameters
PROMPT_DIM = 6 #For 10x10 prompts, padding with ones will be adding for the remaining size
MEM_SIZE = 150  #150 #50 for DUCK-NET, 150 for PraNet
TEMPERATURE= 1    #1, 0.3 for DUCKNET
ADDRESSING_TYPE = "soft" # "soft" or "hard"

#Default Training Parameters
if selected_model == "SwinUNet":
    INPUT_SIZE = 224
elif selected_model == "PraNet":
    INPUT_SIZE = 352
else:
    INPUT_SIZE = 352

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
EPOCHS= 150
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CLASS = 1

# Portable paths (optionally overridable by environment variables)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("APEX_PROJECT_ROOT", SCRIPT_DIR.parents[1]))
DATA_ROOT = Path(os.getenv("APEX_DATA_ROOT", PROJECT_ROOT))
DATASET_ROOT = DATA_ROOT / "dataset_exp"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "polyp"

def resolve_memory_path(model_name: str) -> Path:
    """
    Resolve prompt memory weight path from the outputs folder.
    Priority:
      1) APEX_MEMORY_PATH env var (explicit file)
      2) Any .pth file in outputs containing the model name
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

#TransUNet Parameters
VIT_NAME = "R50-ViT-B_16"
PATCHES_SIZE = 16
config_vit = CONFIGS_ViT_seg[VIT_NAME]
config_vit.n_classes = 1
config_vit.n_skip = 3
config_vit.patches.size = (PATCHES_SIZE, PATCHES_SIZE)
config_vit.patches.grid = (int(INPUT_SIZE/PATCHES_SIZE), int(INPUT_SIZE/PATCHES_SIZE))

#SwinUNet Parameters
config_swin = swin_config()

def worker_init_fn(worker_id): #You can delete it and dataloader parameters for previous experiments except PraNet
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

# Models
models = {
    "PraNet": PraNet(),
    "TransUNet": ViT_seg(config_vit, img_size=INPUT_SIZE, num_classes=N_CLASS),
    "SwinUNet": Swin_ViT_seg(config_swin, img_size=INPUT_SIZE, num_classes=N_CLASS),
}

# Performance Metrics
dice = Dice(average="micro").to(DEVICE) #performance_metrics.DiceScore() #Dice(average="micro").to(DEVICE)
miou = MeanIoU(num_classes=2, include_background=True).to(DEVICE)
precision = Precision(task="binary", average='macro').to(DEVICE)
recall = Recall(task="binary", average='macro').to(DEVICE)


main_dice_list = []
main_iou_list = []
results = []
for target_domain_name in target_domain_names:
    print(f"{target_domain_name} is starting to prepare for testing the memory")


    if target_domain_name == "CVC-ColonDB":
        TEST_IMAGE_PATH = DATASET_ROOT / target_domain_name / "image"
        TEST_MASK_PATH = DATASET_ROOT / target_domain_name / "mask"
    else:
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


    testLoader2 = DataLoader(test_dataset_masks, 
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
        "PraNet":    PRETRAINED_WEIGHTS_DIR / "pretrain-PraNet.pth",
        "TransUNet": PRETRAINED_WEIGHTS_DIR / "pretrain-TransUNet.pth",
        "SwinUNet":  PRETRAINED_WEIGHTS_DIR / "pretrain-SwinUNet.pth",
    }
    PRETRAIN_PATH = PRETRAIN_PATHS[selected_model]

    if selected_model == "TransUNet" or selected_model == "SwinUNet":
        lambda_factor = 0.7
    else:   #PraNet
        lambda_factor = 0.4

    model = models[selected_model].to(DEVICE)

    # Load Pretrain Models
    if selected_model == "PraNet":
        model.load_state_dict(torch.load(PRETRAIN_PATH, weights_only=True, map_location=DEVICE), strict = True)
    else:
        model.load_state_dict(torch.load(PRETRAIN_PATH, weights_only=True, map_location=DEVICE))

    # Load Prompt Memory weights

    # Memory For Transformer Models
    if selected_model == "TransUNet" or selected_model == "SwinUNet":
        prompt_memory = MemoryTransformer(memory_size = MEM_SIZE, prompt_size=PROMPT_DIM, temperature=TEMPERATURE, device=DEVICE, lambda_factor = lambda_factor).to(DEVICE)
        prompt_memory.load_state_dict(torch.load(MEMORY_PATH, weights_only=True), strict=True)    #Memory bank loading
    # Prompt Memory for CNN Models
    else:   
        prompt_memory = MemoryCNN(memory_size = MEM_SIZE, prompt_size=PROMPT_DIM, temperature=TEMPERATURE, device=DEVICE, lambda_factor = lambda_factor).to(DEVICE)
        prompt_memory.load_state_dict(torch.load(MEMORY_PATH, weights_only=True), strict=True)    #Memory bank loading

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
    image_num = 16
    counter = 0
    inference_time_list = []

    with torch.no_grad():
        model.eval()
        prompt_memory.eval()

        for (i, (images, masks, domain_labels, fname)) in enumerate(testLoader):
            images, labels, domain_labels = images.to(DEVICE), masks.to(DEVICE).long(), domain_labels.to(DEVICE)

            prompted_images, addressing_vec, prompt, _ = prompt_memory(images, domain_labels)

            preds = torch.sigmoid(model(prompted_images))
            preds = torch.round(preds)

            # Calculate Performance Metrics
            dice_score = dice(preds, labels.long())
            dice_scores.append(dice_score.cpu().detach().numpy())
            miou_score = miou(preds.long(), labels.to(DEVICE))
            miou_scores.append(miou_score.cpu().detach().numpy())

            total_dice += dice_score /testSteps
            total_miou += miou_score / testSteps
            total_precision += precision(preds, labels.to(DEVICE)) / testSteps
            total_recall += recall(preds, labels.to(DEVICE)) / testSteps         
        
            dice_prompted_image = dice(preds, labels.long().to(DEVICE)).cpu().detach().numpy()
            
    # Convert to numpy
    print("DICE: ", total_dice.cpu().detach().numpy())
    print("mIoU: ", total_miou.cpu().detach().numpy())
    print("Precision: ", total_precision.cpu().detach().numpy())
    print("Recall: ", total_recall.cpu().detach().numpy())
    print("DICE Std.: ", f"{np.std(dice_scores):.6f}")
    print("mIoU Std.: ", f"{np.std(miou_scores):.6f}")

    main_dice_list.append(dice_scores)
    main_iou_list.append(miou_scores)

main_dice_list_x = sum(main_dice_list, [])
main_iou_list_x = sum(main_iou_list, [])

print("Main Mean DICE: ", np.std(main_dice_list_x))
print("Main Std mIoU: ", np.std(main_iou_list_x))