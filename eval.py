import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torchvision.models as models
from torchvision.transforms.functional import resize
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from models.model import VQVAE
from PIL import Image
import os
from utils.utils import get_data_loader, count_parameters, save_img_tensors_as_grid
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import ssl
# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2, data_range=1.0):
    # We'll use SSIM from the torchmetrics library for simplicity
    from torchmetrics.functional import structural_similarity_index_measure
    return structural_similarity_index_measure(img1, img2, data_range=data_range)


# Function to check latent space utilization
def calculate_latent_space_utilization(vqvae, dataloader, device):
    unique_codes = set()
    vqvae.eval()
    
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            # Forward pass through the encoder
            code_indices = vqvae.return_indices(x)
            # Quantize the latent space using the codebook
            # Add the unique indices to the set
            unique_codes.update(code_indices.cpu().numpy().flatten())
    
    utilization = len(unique_codes) / vqvae.num_embeddings
    return utilization


def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Calculate the mean and covariance distance
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Load your trained VQ-VAE model
# vqvae = VQVAE(...)
# vqvae.load_state_dict(torch.load('vqvae.pth'))
vqvae = VQVAE(latent_dim = 3, num_embeddings = 512, beta=0.25, use_ema=False, e_width=64,d_width=64) 
vqvae.to(device)

val_path = '/Users/ayanfe/Documents/Datasets/Waifus/Val'
data_loader = get_data_loader(val_path, batch_size=64, num_samples=10_000)

model_path = 'weights/waifu-vqvae.pth'

# Optionally load model weights if needed
checkpoint = torch.load(model_path, map_location=torch.device(device))
vqvae.load_state_dict(checkpoint['model_state_dict'])

# Load InceptionV3 model for FID calculation using new API
weights = Inception_V3_Weights.DEFAULT
inception_model = inception_v3(weights=weights, transform_input=False)
inception_model = create_feature_extractor(inception_model, return_nodes={'avgpool': 'features'})
inception_model.eval().to(device)

psnr_values = []
ssim_values = []
real_features = []
fake_features = []
vqvae.eval()

# Compute PSNR, SSIM, and extract features for FID
with torch.no_grad():
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        recon_x, _ = vqvae(x)

        # Calculate PSNR and SSIM
        psnr_values.append(calculate_psnr(x, recon_x).item())
        ssim_values.append(calculate_ssim(x, recon_x).item())

        # Resize images to 299x299 for InceptionV3
        x_resized = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        recon_x_resized = F.interpolate(recon_x, size=(299, 299), mode='bilinear', align_corners=False)

        # Extract features from real and generated images
        real_features.append(inception_model(x_resized)['features'].cpu().numpy().reshape(x.size(0), -1))
        fake_features.append(inception_model(recon_x_resized)['features'].cpu().numpy().reshape(x.size(0), -1))

real_features = np.concatenate(real_features, axis=0)
fake_features = np.concatenate(fake_features, axis=0)

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

# Compute Latent Space Utilization
latent_space_utilization = calculate_latent_space_utilization(vqvae, data_loader, device)

# Compute FID
fid_score = calculate_fid(real_features, fake_features)

print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Latent Space Utilization: {latent_space_utilization:.4f}")
print(f"FID Score: {fid_score:.4f}")
