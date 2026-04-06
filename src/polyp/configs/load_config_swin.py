import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.SwinUnet.config import get_config

def swin_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/BKAI/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='BKAI', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=200, help='maximum epoch number to train') #150 2
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch_size per gpu')  #24
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='segmentation network learning rate') #0.01
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=42, help='random seed') #1234
    parser.add_argument('--cfg', type=str, default = "/mnt/disk1/hjlee/GenTransImgSeg_Thesis/models/SwinUnet/configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', )
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

    return config