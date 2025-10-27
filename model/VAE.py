from abc import abstractmethod, ABC

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import pytorch_lightning as pl


class VAEAnomalyDetection(pl.LightningModule, ABC):
    """
    Variational Autoencoder (VAE) for anomaly detection. The model learns a low-dimensional representation of the input
    data using an encoder-decoder architecture, and uses the learned representation to detect anomalies.

    The model is trained to minimize the Kullback-Leibler (KL) divergence between the learned distribution of the latent
    variables and the prior distribution (a standard normal distribution). It is also trained to maximize the likelihood
    of the input data under the learned distribution.

    This implementation uses PyTorch Lightning to simplify training and improve reproducibility.
    """

    def __init__(self, input_channels: int, image_size: int, latent_size: int, L: int = 10, lr: float = 1e-3, log_steps: int = 1_000):
        super().__init__()
        self.save_hyperparameters()
        self.L = L
        self.lr = lr
        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_size = latent_size

        # Calculate the flattened size after convolutional layers for the encoder's linear part
        # Assuming a 4-layer conv net with stride 2 and padding 1, the spatial dimension is image_size / (2^4) = image_size / 16
        # And assuming the last conv layer outputs 128 channels.
        final_spatial_dim = self.image_size // (2**4)
        self.final_conv_output_dim = 128 * final_spatial_dim * final_spatial_dim # 128 is example output channels of last conv

        self.encoder = self.make_encoder(input_channels, latent_size, image_size)
        self.decoder = self.make_decoder(latent_size, input_channels, image_size)
        self.prior = Normal(0, 1)
        self.log_steps = log_steps

    @abstractmethod
    def make_encoder(self, input_channels: int, latent_size: int, image_size: int) -> nn.Module:
        """
        Abstract method to create the encoder network.

        Args:
            input_size (int): Number of input features.
            latent_size (int): Size of the latent space.

        Returns:
            nn.Module: Encoder network.
        """
        pass

    @abstractmethod
    def make_decoder(self, latent_size: int, output_channels: int, image_size: int) -> nn.Module:
        """
        Abstract method to create the decoder network.

        Args:
            latent_size (int): Size of the latent space.
            output_size (int): Number of output features.

        Returns:
            nn.Module: Decoder network.
        """
        pass

    def forward(self, x: torch.Tensor) -> dict:
        """
        Computes the forward pass of the model and returns the loss and other relevant information.

        Args:
            x (torch.Tensor): Input data. Shape [batch_size, num_channels, height, width].

        Returns:
            Dictionary containing:
            - loss: Total loss.
            - kl: KL-divergence loss.
            - recon_loss: Reconstruction loss.
            - recon_mu: Mean of the reconstructed input.
            - recon_sigma: Standard deviation of the reconstructed input.
            - latent_dist: Distribution of the latent space.
            - latent_mu: Mean of the latent space.
            - latent_sigma: Standard deviation of the latent space.
            - z: Sampled latent space.

        """
        pred_result = self.predict(x)

        log_lik_per_sample = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x.unsqueeze(0))
        log_lik = log_lik_per_sample.mean(dim=0).sum() # Average over L, sum over C, H, W

        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def predict(self, x) -> dict:
        """
        Compute the output of the VAE. Does not compute the loss compared to the forward method.

        Args:
            x: Input tensor of shape [batch_size, input_channels, height, width].

        Returns:
            Dictionary containing:
            - latent_dist: Distribution of the latent space.
            - latent_mu: Mean of the latent space.
            - latent_sigma: Standard deviation of the latent space.
            - recon_mu: Mean of the reconstructed input.
            - recon_sigma: Standard deviation of the reconstructed input.
            - z: Sampled latent space.

        """
        # Ensure x is a tensor and has the correct 2D shape for convolutions
        x = x[0] if isinstance(x, (list, tuple)) else x
        # No flattening needed here, input is already [B, C, H, W]

        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1) #both with size [batch_size, latent_size]
        latent_sigma = softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        # z = z.view(self.L * batch_size, self.latent_size) # No longer need to flatten z after convolution encoder
        z = z.reshape(self.L * batch_size, self.latent_size)

        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        # recon_mu and recon_sigma are now [L * batch_size, C*2, H, W] -> needs to be reshaped
        # Reshape recon_mu/sigma back to [L, batch_size, C, H, W]
        recon_mu = recon_mu.view(self.L, batch_size, self.input_channels, self.image_size, self.image_size)
        recon_sigma = recon_sigma.view(self.L, batch_size, self.input_channels, self.image_size, self.image_size)

        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Determines if input samples are anomalous based on a given threshold.
        
        Args:
            x: Input tensor of shape (batch_size, num_features).
            alpha: Anomaly threshold. Values with probability lower than alpha are considered anomalous.
        
        Returns:
            A binary tensor of shape (batch_size,) where `True` represents an anomalous sample and `False` represents a 
            normal sample.
        """
        p = self.reconstructed_probability(x)
        return p < alpha

    def reconstructed_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability density of the input samples under the learned
        distribution of reconstructed data.

        Args:
            x: Input data tensor of shape (batch_size, num_channels, height, width).

        Returns:
            A tensor of shape (batch_size,) containing the probability densities of
            the input samples under the learned distribution of reconstructed data.
        """
        # Ensure x is a tensor and has the correct 2D shape for convolutions
        x = x[0] if isinstance(x, (list, tuple)) else x
        # No flattening needed here, input is already [B, C, H, W]

        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])

        if torch.isnan(pred['recon_sigma']).any() or torch.isinf(pred['recon_sigma']).any():
            print(f"  WARNING: recon_sigma contains NaN or Inf values!")

        log_prob_per_pixel = recon_dist.log_prob(x.unsqueeze(0)) # Shape [L, B, C, H, W]
        
        # Calculate average likelihood per pixel across L, C, H, W
        # First, convert log_prob to actual probabilities per pixel (still L, B, C, H, W)
        probabilities_per_pixel = log_prob_per_pixel.exp()

        # Then, average these probabilities over L, C, H, W
        # This gives us a single average probability density value per image in the batch
        p = probabilities_per_pixel.mean(dim=0).mean(dim=[1, 2, 3]) # Result: [B]

        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generates a batch of samples from the learned prior distribution.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            A tensor of shape (batch_size, num_features) containing the generated
            samples.
        """
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)
    
    
    def training_step(self, batch, batch_idx):
        # Unpack batch (e.g., from MNIST: (images, targets)) and ensure 2D shape for convolutions
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.forward(x)
        if self.global_step % self.log_steps == 0:
            self.log('train/loss', loss['loss'])
            self.log('train/loss_kl', loss['kl'], prog_bar=False)
            self.log('train/loss_recon', loss['recon_loss'], prog_bar=False)
            self._log_norm()

        return loss
    

    def validation_step(self, batch, batch_idx):
        # Unpack batch and ensure 2D shape for convolutions
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.forward(x)
        self.log('val/loss', loss['loss'], on_epoch=True, prog_bar=False)
        self.log('val/kl', loss['kl'], on_epoch=True, prog_bar=False)
        self.log('val/recon_loss', loss['recon_loss'], on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def _log_norm(self):
        norm1 = sum(p.norm(1) for p in self.parameters())
        norm1_grad = sum(p.grad.norm(1) for p in self.parameters() if p.grad is not None)
        self.log('norm1_params', norm1)
        self.log('norm1_grad', norm1_grad)

class VAEAnomalyTabular(VAEAnomalyDetection):

    # Redefine __init__ to match the new VAEAnomalyDetection signature
    def __init__(self, input_channels: int, image_size: int, latent_size: int, L: int = 10, lr: float = 1e-3, log_steps: int = 1_000):
        super().__init__(input_channels, image_size, latent_size, L, lr, log_steps)

    def make_encoder(self, input_channels: int, latent_size: int, image_size: int):
        # Convolutional Encoder architecture
        final_spatial_dim = image_size // (2**4) # = 8 for image_size=128
        final_conv_output_channels = 128 # Output channels for the last conv layer before flattening
        
        return nn.Sequential(
            # Input: (batch_size, input_channels, image_size, image_size) e.g., (B, 1, 128, 128)
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1), # -> (B, 16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, final_conv_output_channels, kernel_size=4, stride=2, padding=1), # -> (B, 128, 8, 8)
            nn.ReLU(),
            nn.Flatten(), # -> (B, 128*8*8) = (B, 8192)
            nn.Linear(final_conv_output_channels * final_spatial_dim * final_spatial_dim, latent_size * 2) # -> (B, latent_size * 2)
        )

    def make_decoder(self, latent_size: int, output_channels: int, image_size: int):
        # Convolutional Decoder architecture
        final_spatial_dim = image_size // (2**4) # = 8 for image_size=128
        final_conv_output_channels = 128
        
        return nn.Sequential(
            nn.Linear(latent_size, final_conv_output_channels * final_spatial_dim * final_spatial_dim), # -> (B, 8192)
            nn.ReLU(),
            nn.Unflatten(1, (final_conv_output_channels, final_spatial_dim, final_spatial_dim)), # -> (B, 128, 8, 8)
            nn.ConvTranspose2d(final_conv_output_channels, 64, kernel_size=4, stride=2, padding=1), # -> (B, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # -> (B, 16, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels * 2, kernel_size=4, stride=2, padding=1) # -> (B, output_channels*2, 128, 128)
        )


