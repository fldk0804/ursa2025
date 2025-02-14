import argparse
import json
import os

def parse_your_arguments():
    parser = argparse.ArgumentParser(description="Training script for ML experiments")

    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use: PAMAP2, AnotherDataset, etc.")
    parser.add_argument("--model", type=str, required=True, help="Model type: ViT, CNN, etc.")
    parser.add_argument("--framework", type=str, required=True, help="Framework: Supervised, SimCLR, MAE, etc.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--experiment_tag", type=str, default="default_exp", help="Experiment tag")

    args = parser.parse_args()
    return args

def setup_experiment(args):
    """Creates an experiment directory and saves config files."""
    exp_dir = f"experiment/expX_{args.dataset}_{args.model}_{args.framework}_{args.experiment_tag}"

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    args.exp_dir = exp_dir  # Store path in args

    # Save experiment configuration
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return args
