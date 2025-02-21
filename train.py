import argparse
import logging
import os
import random
import numpy np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')  # Optimized for GTX 1070 with mixed precision
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


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

    # Update path handling for Windows compatibility
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Update dataset config paths
    dataset_config = {
        'Synapse': {
            'root_path': os.path.join(base_path, 'data', 'Synapse', 'train_npz'),
            'list_dir': os.path.join(os.path.dirname(__file__), 'lists', 'lists_Synapse'),  # Changed this line
            'num_classes': 9,
        },
    }
    
    # Update args with proper paths
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = os.path.normpath(dataset_config[dataset_name]['root_path'])
    args.list_dir = os.path.normpath(dataset_config[dataset_name]['list_dir'])
    args.is_pretrain = True
    
    # Construct model save path
    args.exp = f'TU_{dataset_name}{args.img_size}'
    model_components = [
        'TU_pretrain' if args.is_pretrain else 'TU',
        args.vit_name,
        f'skip{args.n_skip}',
        f'vitpatch{args.vit_patches_size}' if args.vit_patches_size != 16 else None,
        f'epo{args.max_epochs}' if args.max_epochs != 30 else None,
        f'bs{args.batch_size}',
        f'lr{args.base_lr}' if args.base_lr != 0.01 else None,
        f'{args.img_size}',
        f's{args.seed}' if args.seed != 1234 else None
    ]
    
    # Filter out None values and join components
    model_name = '_'.join(filter(None, model_components))
    snapshot_path = os.path.join(base_path, 'model', args.exp, model_name)
    
    # Create directories if they don't exist
    os.makedirs(snapshot_path, exist_ok=True)
    print(f"Models will be saved to: {snapshot_path}")
    
    # Debugging: Print paths to verify
    print(f"Root path: {args.root_path}")
    print(f"List dir: {args.list_dir}")
    train_list_path = os.path.join(args.list_dir, 'train.txt')
    print(f"Train list path: {train_list_path}")
    
    # Check if train.txt exists
    if not os.path.exists(train_list_path):
        raise FileNotFoundError(f"Train list file not found: {train_list_path}")
    
    # Configure ViT model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                 int(args.img_size / args.vit_patches_size))
    
    # Initialize and load model
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    # Enable mixed precision training
    scaler = GradScaler()

    # Start training
    trainer = {'Synapse': trainer_synapse}
    try:
        result = trainer[dataset_name](args, net, snapshot_path, scaler)
        print(f"\nTraining result: {result}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")