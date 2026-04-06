import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse, trainer_bkai_, trainer_optic_
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/OPTIC/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='BKAI', help='experiment_name')        #OPTIC
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')    #3
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train') #150 2
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')  #24
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate') #0.01
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')  #224
parser.add_argument('--seed', type=int,
                    default=42, help='random seed') #1234
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# parser.add_argument("--dataset_name", default="datasets")
parser.add_argument("--n_class", default=1, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': f'./lists/{args.dataset}',
            'num_classes': args.n_class,
        },
        'BKAI': {
            'num_classes': 1,
        },
        'OPTIC': {
            'num_classes': 3,
        },
    }

    DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    #args.root_path = dataset_config[dataset_name]['root_path']
    #args.list_dir = dataset_config[dataset_name]['list_dir']
    print(f"\n\nDataset: {dataset_name}, Root Path: {args.root_path}, List Dir: {args.list_dir}, Num Classes: {args.num_classes}\n\n")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(DEVICE)
    net.load_from(config)

    # trainer = {'Synapse': trainer_synapse}
    trainer_bkai_(args, net, args.output_dir, DEVICE)
    #trainer_optic_(args, net, args.output_dir, DEVICE)


# python train.py --output_dir ./model_out/datasets/drishti_gs --dataset datasets --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split