import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

class WorkerInitializer:
    def __init__(self, seed):
        self.seed = seed
    
    def __call__(self, worker_id):
        random.seed(self.seed + worker_id)

def trainer_synapse(args, model, snapshot_path, scaler):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    
    # Create log directory
    log_path = os.path.join(snapshot_path, "log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Training parameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Initialize dataset
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ])
    )
    print(f"The length of train set is: {len(db_train)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Initialize dataloader with Windows-specific settings
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 8,  # Reduce workers on Windows
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # Model setup
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # Loss and optimizer
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Setup tensorboard
    log_dir = os.path.join(snapshot_path, 'log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training loop setup
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    
    # Training loop
    try:
        for epoch_num in tqdm(range(max_epoch), ncols=70):
            for i_batch, sampled_batch in enumerate(trainloader):
                # Forward pass with mixed precision
                with autocast():
                    outputs = model(sampled_batch['image'].cuda())
                    loss_ce = ce_loss(outputs, sampled_batch['label'].cuda())
                    loss_dice = dice_loss(outputs, sampled_batch['label'].cuda())
                    loss = 0.5 * loss_ce + 0.5 * loss_dice

                # Backward pass with mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                iter_num += 1
                writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                writer.add_scalar('info/loss_dice', loss_dice, iter_num)

                logging.info(f'iteration {iter_num} : loss : {loss.item()}')

                if iter_num % 20 == 0:
                    image = sampled_batch['image'][1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    writer.add_image('train/GroundTruth', sampled_batch['label'][1, ...].unsqueeze(0) * 50, iter_num)

                if iter_num >= max_iterations:
                    break

            if iter_num >= max_iterations:
                break

            # Save checkpoints
            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % 50 == 0:
                save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")

        # Save final model
        save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info(f"save model to {save_mode_path}")

    except KeyboardInterrupt:
        save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_interrupted.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info(f"Training interrupted. Model saved to {save_mode_path}")
        return "Training interrupted and saved!"

    writer.close()
    return "Training Finished!"