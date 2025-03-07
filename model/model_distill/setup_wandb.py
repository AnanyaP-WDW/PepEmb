#!/usr/bin/env python
import os
import argparse
import subprocess
import sys

def check_wandb_installed():
    """Check if wandb is installed"""
    try:
        import wandb
        return True
    except ImportError:
        return False

def install_wandb():
    """Install wandb package"""
    print("Installing Weights & Biases...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    print("Weights & Biases installed successfully!")

def login_to_wandb(api_key=None):
    """Login to wandb"""
    import wandb
    
    if api_key:
        # Login with provided API key
        wandb.login(key=api_key)
        print("Logged in to Weights & Biases successfully!")
    else:
        # Interactive login
        print("\nTo use Weights & Biases:")
        print("1. Create a free account at https://wandb.ai if you don't have one")
        print("2. Get your API key from https://wandb.ai/settings")
        print("3. Enter your API key below or run wandb login in your terminal\n")
        
        wandb.login()
        print("\nLogged in to Weights & Biases successfully!")

def setup_wandb(api_key=None):
    """Setup wandb for monitoring"""
    if not check_wandb_installed():
        install_wandb()
    
    # Import again after installation
    import wandb
    
    # Login to wandb
    login_to_wandb(api_key)
    
    # Print instructions
    print("\n===== Weights & Biases Setup Complete =====")
    print("\nTo monitor your training with W&B, run:")
    print("python model_distill/distill.py --use_wandb [other args...]")
    print("\nYou can view your runs at: https://wandb.ai/dashboard")
    print("\nAdditional W&B features:")
    print("- Hyperparameter sweeps: https://docs.wandb.ai/guides/sweeps")
    print("- Experiment comparison: https://docs.wandb.ai/guides/track/compare-runs")
    print("- Model visualization: https://docs.wandb.ai/guides/track/visualizations")

def init_wandb_test_run():
    """Initialize a test run to verify wandb is working"""
    import wandb
    
    print("Initializing a test W&B run...")
    run = wandb.init(project="protein-distillation-test", name="test-run")
    
    # Log some sample metrics
    for i in range(10):
        wandb.log({
            "loss": 1.0 - i * 0.1,
            "accuracy": i * 0.1
        })
    
    run.finish()
    print("Test run completed! Check your W&B dashboard at https://wandb.ai/dashboard")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Weights & Biases for monitoring")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Your W&B API key")
    parser.add_argument("--test", action="store_true",
                        help="Run a test to verify W&B is working")
    
    args = parser.parse_args()
    
    setup_wandb(args.api_key)
    
    if args.test:
        init_wandb_test_run() 