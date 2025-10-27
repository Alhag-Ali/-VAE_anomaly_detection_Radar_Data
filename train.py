import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import yaml
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model.VAE import VAEAnomalyTabular
from dataset import rand_dataset, mnist_dataset, radar_dataset

from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

ROOT = Path(__file__).parent
SAVED_MODELS = ROOT / 'saved_models'


def make_folder_run() -> Path:
    """
    Get the folder where to store the experiment. 
    The folder is named with the current date and time.
    
    Returns:
        Path: the path to the folder where to store the experiment
    """
    checkpoint_folder = SAVED_MODELS / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_folder.makedirs_p()
    return checkpoint_folder


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-size', '-l', type=int, default=32, dest='latent_size', help='Size of the latent space') # required=True
    parser.add_argument('--num-resamples', '-L', type=int, dest='num_resamples', default=10,
                        help='Number of resamples in the latent distribution during training')
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', default=3, help='Number of epochs to train for')
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=32)
    parser.add_argument('--device', '-d', '--accelerator', type=str, dest='device', default='cpu', help='Device to use for training. Can be cpu, gpu or tpu', choices=['cpu', 'gpu', 'tpu'])
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3, help='Learning rate')
    parser.add_argument('--no-progress-bar', action='store_true', dest='no_progress_bar')
    parser.add_argument('--steps-log-loss', type=int, dest='steps_log_loss', default=1_000, help='Number of steps between each loss logging')
    parser.add_argument('--steps-log-norm-params', type=int, 
                        dest='steps_log_norm_params', default=1_000, help='Number of steps between each model parameters logging')

    return parser.parse_args()


def main():
    """
    Main function to train the VAE model
    """
    args = get_args()
    print(args)
    experiment_folder = make_folder_run()

    # copy model folder into experiment folder
    ROOT.joinpath('model').copytree(experiment_folder / 'model')

    with open(experiment_folder / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    # Initialize model with 2D parameters
    model = VAEAnomalyTabular(input_channels=1, image_size=128, latent_size=args.latent_size, L=args.num_resamples, lr=args.lr)

    transformls = ToTensor()
    train_set, val_set, test_set = radar_dataset(image_size=128)
    
    train_dloader = DataLoader(train_set, batch_size=args.batch_size)

    val_dloader = DataLoader(val_set, args.batch_size)

    checkpoint = ModelCheckpoint(
        dirpath=experiment_folder,
        filename='{epoch:02d}-{val/loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_last=True,
    )

    trainer = Trainer(callbacks=[checkpoint], num_sanity_val_steps=2, limit_val_batches=2, max_epochs=args.epochs)
    trainer.fit(model, train_dloader, val_dloader)


if __name__ == '__main__':
    main()