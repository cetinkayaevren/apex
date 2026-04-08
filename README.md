# From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation

Official Implementation for "From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation" [CVPR 2026 Findings]

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Download Weights](#download-weights)
   - [Polyp Segmentation Weights](#polyp-segmentation-weights)
   - [Optic Disc/Cup Segmentation Weights](#optic-disccup-segmentation-weights)
4. [Prepare Datasets](#prepare-datasets)
5. [Testing (Polyp Segmentation)](#testing-polyp-segmentation)
6. [Testing (Optic Disc/Cup Segmentation)](#testing-optic-disccup-segmentation)
7. [Environment Variables (Optional)](#environment-variables-optional)

---

## Project Structure

```
apex/
├── requirements.txt
├── README.md
├── dataset_exp/                          # Datasets (see below)
│   ├── CVC-ClinicDB_0_0/                 # Polyp datasets
│   ├── Kvasir-SEG_3_3/
│   ├── ETIS-LaribPolypDB/
│   ├── CVC-ColonDB/
│   └── Fundus/                           # Optic datasets
│       ├── REFUGE_0/
│       ├── REFUGE_Valid_3/
│       ├── Drishti_GS_0/
│       └── RIM_ONE_r3/
├── outputs/
│   ├── polyp/
│   │   ├── pretrained_weights/           # Polyp pretrained segmentation weights
│   │   └── memory_weights/               # Polyp prompt memory weights
│   └── optic/
│       ├── pretrained_weights/           # Optic pretrained segmentation weights
│       └── memory_weights/               # Optic prompt memory weights
├── models/                               # Model architectures (source code)
├── loss/                                 # Loss functions
└── src/
    ├── polyp/
    │   ├── memory_prompt_testing.py       # Polyp testing script
    │   ├── memory_prompt_training.py      # Polyp training script
    │   ├── memory/                        # Memory modules
    │   ├── dataloader/                    # Data loaders
    │   ├── prompt/                        # Visual prompt modules
    │   └── configs/                       # Model configs
    └── optic/
        ├── memory_prompt_testing.py       # Optic testing script
        ├── memory_storage_training.py     # Optic training script
        ├── memory/                        # Memory modules
        └── dataloader/                    # Data loaders
```

---

## Environment Setup

**Prerequisites:** Python 3.10+, CUDA-capable GPU (recommended)

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the environment:**

   ```bash
   # Windows
   venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA** (adjust the CUDA version to match your system):

   Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and follow the instructions for your platform. For example:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install remaining dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Download Weights

Download the pretrained segmentation weights and the trained prompt memory weights from the links below and place them in the corresponding folders.

### Polyp Segmentation Weights

#### Pretrained Segmentation Weights (Polyp)

| Model | Download Link | Place in |
|---|---|---|
| PraNet | [link](https://drive.google.com/file/d/1NC9He4_nUiwBA8ZHY61Utrhh5XG6QFHa/view?usp=sharing) | `outputs/polyp/pretrained_weights/pretrain-PraNet.pth` |
| TransUNet | [link](https://drive.google.com/file/d/1gY0gaXqbOWQeAnaFaxvnf20K0iS1DLB-/view?usp=sharing) | `outputs/polyp/pretrained_weights/pretrain-TransUNet.pth` |
| SwinUNet | [link](https://drive.google.com/file/d/1Oa91aqcZndY_s5zmbGEwRcJMFwaVIpPx/view?usp=sharing) | `outputs/polyp/pretrained_weights/pretrain-SwinUNet.pth` |

#### Prompt Memory Weights (Polyp)

| Model | Download Link | Place in |
|---|---|---|
| PraNet | [link](https://drive.google.com/file/d/1o-2zHhLWBXt6vJgDNeWtkOF1ByxYZhTA/view?usp=sharing) | `outputs/polyp/memory_weights/` |
| TransUNet | [link](https://drive.google.com/file/d/1EWf5MlhtcEAW82iVoDGlC1XZ8vMul253/view?usp=sharing) | `outputs/polyp/memory_weights/` |
| SwinUNet | [link](https://drive.google.com/file/d/1IBJvg_J6t3H17IOxh6bM4JsO3fPZEQhd/view?usp=sharing) | `outputs/polyp/memory_weights/` |

> **Naming convention for memory weights:** The testing script auto-detects `.pth` files inside the `memory_weights/` folder by matching the model name in the filename. Files with `best` or `last` in the name are preferred.

After downloading, your polyp weights folder should look like:

```
outputs/polyp/
├── pretrained_weights/
│   ├── pretrain-PraNet.pth
│   ├── pretrain-TransUNet.pth
│   └── pretrain-SwinUNet.pth
└── memory_weights/
    ├── prompt_memory_polyp_PraNet_..._best.pth
    ├── prompt_memory_polyp_TransUNet_..._best.pth
    └── prompt_memory_polyp_SwinUNet_..._best.pth
```

### Optic Disc/Cup Segmentation Weights

#### Pretrained Segmentation Weights (Optic)

| Model | Download Link | Place in |
|---|---|---|
| UNet | [link](https://drive.google.com/file/d/1p5fF3-fhIkgkwfbRs3tEMouUKKMOHLXm/view?usp=sharing) | `outputs/optic/pretrained_weights/pretrain-UNet.pth` |
| ResUnet | [link](https://drive.google.com/file/d/1j3hRT7KbMak8JHKYnA6XnSOxPKGhkj2Z/view?usp=sharing) | `outputs/optic/pretrained_weights/pretrain-ResUnet.pth` |
| TransUNet | [link](https://drive.google.com/file/d/1zINhiZyIPuuGjXR5wpBHfS8RV9ijILl4/view?usp=sharing) | `outputs/optic/pretrained_weights/pretrain-TransUNet.pth` |
| SwinUNet | [link](https://drive.google.com/file/d/1G7Mp72vAWxD3hMG1yaJpzOzrJvNVFUuV/view?usp=sharing) | `outputs/optic/pretrained_weights/pretrain-SwinUNet.pth` |

#### Prompt Memory Weights (Optic)

| Model | Download Link | Place in |
|---|---|---|
| UNet | [link](https://drive.google.com/file/d/1QIo0rCI3HB0sEl7FtzgIREO38oUSiBCg/view?usp=sharing) | `outputs/optic/memory_weights/` |
| ResUnet | [link](https://drive.google.com/file/d/1G7JLFTUujYmZezRsnjBudvIucWRMP74M/view?usp=sharing) | `outputs/optic/memory_weights/` |
| TransUNet | [link](https://drive.google.com/file/d/18LiI1jUdVgak-x-TVwPJ2BVIyjFQU2Tk/view?usp=sharing) | `outputs/optic/memory_weights/` |
| SwinUNet | [link](https://drive.google.com/file/d/1kYoORGdPnvXYM0E8t0g88Kb7e26p2f3d/view?usp=sharing) | `outputs/optic/memory_weights/` |

After downloading, your optic weights folder should look like:

```
outputs/optic/
├── pretrained_weights/
│   ├── pretrain-UNet.pth
│   ├── pretrain-ResUnet.pth
│   ├── pretrain-TransUNet.pth
│   └── pretrain-SwinUNet.pth
└── memory_weights/
    ├── prompt_memory_optic_UNet_..._best.pth
    ├── prompt_memory_optic_ResUnet_..._best.pth
    ├── prompt_memory_optic_TransUNet_..._best.pth
    └── prompt_memory_optic_SwinUNet_..._best.pth
```

---

## Prepare Datasets

Download the datasets and place them under the `dataset_exp/` folder at the project root.

### Polyp Segmentation Datasets

```
dataset_exp/
├── CVC-ClinicDB_0_0/
│   └── test/
│       ├── image/
│       └── mask/
├── Kvasir-SEG_3_3/
│   └── test/
│       ├── image/
│       └── mask/
├── ETIS-LaribPolypDB/
│   └── test/
│       ├── image/
│       └── mask/
└── CVC-ColonDB/            # No test/ subfolder for this dataset
    ├── image/
    └── mask/
```

### Optic Disc/Cup Segmentation Datasets

```
dataset_exp/Fundus/
├── REFUGE_0/
│   └── test/
│       ├── image/
│       └── mask/
├── REFUGE_Valid_3/
│   └── test/
│       ├── image/
│       └── mask/
├── Drishti_GS_0/
│   └── test/
│       ├── image/
│       └── mask/
└── RIM_ONE_r3/
    └── test/
        ├── image/
        └── mask/
```

---

## Testing (Polyp Segmentation)

Run the testing script from the `src/polyp/` directory. Select the model with the `--model` flag.

**Available models:** `PraNet`, `TransUNet`, `SwinUNet`

```bash
cd src/polyp

# Test with PraNet (default)
python memory_prompt_testing.py --model PraNet

# Test with TransUNet
python memory_prompt_testing.py --model TransUNet

# Test with SwinUNet
python memory_prompt_testing.py --model SwinUNet
```

The script will automatically:
- Load the pretrained segmentation weights from `outputs/polyp/pretrained_weights/`
- Load the prompt memory weights from `outputs/polyp/memory_weights/`
- Evaluate on all four target domains: CVC-ClinicDB, Kvasir-SEG, ETIS-LaribPolypDB, CVC-ColonDB
- Print Dice, mIoU, Precision, and Recall scores for each domain

---

## Testing (Optic Disc/Cup Segmentation)

Run the testing script from the `src/optic/` directory. Select the model with the `--model` flag.

**Available models:** `UNet`, `ResUnet`, `TransUNet`, `SwinUNet`

```bash
cd src/optic

# Test with UNet (default)
python memory_prompt_testing.py --model UNet

# Test with ResUnet
python memory_prompt_testing.py --model ResUnet

# Test with TransUNet
python memory_prompt_testing.py --model TransUNet

# Test with SwinUNet
python memory_prompt_testing.py --model SwinUNet
```

The script will automatically:
- Load the pretrained segmentation weights from `outputs/optic/pretrained_weights/`
- Load the prompt memory weights from `outputs/optic/memory_weights/`
- Evaluate on all four target domains: REFUGE, REFUGE_Valid, Drishti_GS, RIM_ONE_r3
- Print Dice, mIoU, Precision, and Recall scores for each domain

---

## Environment Variables (Optional)

You can override the default paths using environment variables if your data or weights are stored elsewhere:

| Variable | Description | Default |
|---|---|---|
| `APEX_PROJECT_ROOT` | Project root directory | Auto-detected from script location |
| `APEX_DATA_ROOT` | Root folder containing `dataset_exp/` | Same as project root |
| `APEX_MODEL` | Model name (polyp: `PraNet`, `TransUNet`, `SwinUNet`; optic: `UNet`, `ResUnet`, `TransUNet`, `SwinUNet`) | `PraNet` (polyp) / `UNet` (optic) |
| `APEX_MEMORY_PATH` | Explicit path to a memory weight file | Auto-detected from `outputs/<task>/memory_weights/` |

Example:

```bash
# Linux / macOS
export APEX_DATA_ROOT=/path/to/your/data
python memory_prompt_testing.py --model TransUNet

# Windows
set APEX_DATA_ROOT=C:\path\to\your\data
python memory_prompt_testing.py --model TransUNet
```
