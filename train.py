import os
from utils.parse_utils import parse_your_arguments, setup_experiment
from utils.log import setup_logger
from data.dataloaders import create_dataloader
from models.model_utils import init_backbone_model
from loss.YOURLoss import init_loss_func
from frameworks.supervised import supervised_train

# Step 1: Parse arguments
args = parse_your_arguments()

# Step 2: Setup experiment directory & config
args = setup_experiment(args)

# Step 3: Initialize logger
logger = setup_logger(args.exp_dir)

logger.info("Experiment Initialized")
logger.info(f"Dataset: {args.dataset}, Model: {args.model}, Framework: {args.framework}, Batch Size: {args.batch_size}")

# Step 4: Create DataLoaders
train_dataloader = create_dataloader(args, "train", args.batch_size)
val_dataloader = create_dataloader(args, "val", args.batch_size)

# Step 5: Initialize Model & Loss
backbone_model = init_backbone_model(args)
loss_func = init_loss_func(args)

# Step 6: Start Training
logger.info("Starting training...")
supervised_train(args, backbone_model, loss_func, train_dataloader, val_dataloader)
logger.info("Training completed!")
