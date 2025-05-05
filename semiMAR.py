import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
import re
import nibabel as nib
from skimage.transform import resize  # Or use cv2

# --- Configuration ---
IMG_SIZE = 256  # Example image size. (ACDNet used 416x416. 512x512 is also an option
INPUT_CHANNELS = 1  # Grayscale CT images
OUTPUT_CHANNELS = 1
LR_G = 1e-4  # Learning rate for generator (SemiMAR paper: 0.0001)
LR_D = 1e-4  # Learning rate for discriminator (SemiMAR paper: 0.0001)
BETAS = (0.5, 0.999)  # Adam optimizer betas (SemiMAR paper uses these)
EPOCHS = 60  # SemiMAR paper uses 60
LR_DECAY_EPOCH = 20  # SemiMAR paper halves every 20 epochs
BATCH_SIZE = 16  # Adjust based on GPU memory (Paper doesn't specify, start small)


# --- Data Paths ---
DATA_DIR_XP = '/home/mcosta/PycharmProjects/Semi_ACDNet/clinic_metal_results_copy_mode/'  # Directory with Xp .npy files
DATA_DIR_LOW = 'data/test/CTPelvic1K_METAL/'  # Directory with corresponding Xlow .nii.gz files
DATA_DIR_HIGH = 'data/test/CTPelvic1K_METAL_FREE'  # Directory with unpaired Xhigh .nii.gz files
# --- End Data Paths ---
CHECKPOINT_DIR = './checkpoints_semimar'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights (from SemiMAR paper)
# Note: Paper uses different weights for simulated vs clinical. Using clinical weights here.
lambda_cyc = 20.0
lambda_art = 20.0
lambda_model_error = 20.0
lambda_im = 1.0  # Image prior loss weight for clinical data
lambda_cont = 1.0  # Contrastive loss weight for clinical data

# Preprocessing Params
SLICE_AXIS = 2  # Axis for slicing NIfTI files
WINDOW_MIN = -1000  # HU window min for X_low/X_high
WINDOW_MAX = 1000  # HU window max for X_low/X_high


# --- Model Architecture (Keep Generator, Discriminator, ResidualBlock as before) ---

# Residual Block (Helper)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


# Encoder (NetE) - Based on SemiMAR figure
class EncoderNet(nn.Module):
    # Assuming similar structure to ACDNet's encoder based on diagram
    # C32K7S1P3 C64K4S2P1 C128K4S2P1 + ResBlocks
    def __init__(self, input_nc=INPUT_CHANNELS, ngf=32, n_blocks=4):
        super(EncoderNet, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        in_features = ngf;
        out_features = ngf * 2  # 64
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features;
        out_features = in_features * 2  # 128
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        self.initial_layers = nn.Sequential(*model)

        res_blocks = []
        for _ in range(n_blocks):
            res_blocks += [ResidualBlock(out_features)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)

    def forward(self, input):
        initial_out = self.initial_layers(input)
        res_out = self.res_blocks_seq(initial_out)
        return res_out  # Features for contrastive loss


# Decoder (NetD) - Based on SemiMAR figure
class DecoderNet(nn.Module):
    # Symmetrical to Encoder: ResBlocks + C128K5S1P2 C64K5S1P2 C1K7S1P3
    # Assuming K5S1P2 means ConvTranspose with k=3, s=2, p=1, op=1 (standard upsampling)
    def __init__(self, output_nc=OUTPUT_CHANNELS, ngf=128, n_blocks=4):
        super(DecoderNet, self).__init__()
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks += [ResidualBlock(ngf)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)

        model = []
        in_features = ngf;
        out_features = in_features // 2  # 64
        model += [
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features;
        out_features = in_features // 2  # 32
        model += [
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  # Output normalized to [-1, 1]
        ]
        self.upsample_layers = nn.Sequential(*model)

    def forward(self, input):
        res_out = self.res_blocks_seq(input)
        output = self.upsample_layers(res_out)
        return output


# Generator combining Encoder and Decoder
class Generator(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, output_nc=OUTPUT_CHANNELS, ngf_enc=32, ngf_dec_in=128, n_blocks=4):
        super(Generator, self).__init__()
        self.encoder = EncoderNet(input_nc, ngf_enc, n_blocks)
        self.decoder = DecoderNet(output_nc, ngf_dec_in, n_blocks)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output, features  # Return features for contrastive loss


# Discriminator (PatchGAN) - Same as before
class Discriminator(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        kw = 4;
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult;
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, inplace=True)
            ]
        nf_mult_prev = nf_mult;
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, inplace=True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# --- Utility Functions for Dataset ---

def preprocess_nifti_slice(slice_data, target_size, win_min, win_max):
    """Applies windowing, normalization [-1,1], and resizing to a NIfTI slice."""
    # 1. Apply Windowing
    img_slice_windowed = np.clip(slice_data, win_min, win_max)
    # 2. Normalize to [-1, 1]
    if win_max > win_min:
        img_slice_normalized = 2.0 * (img_slice_windowed - win_min) / (win_max - win_min) - 1.0
    else:
        img_slice_normalized = np.zeros_like(img_slice_windowed)
    # 3. Resize
    img_resized = resize(img_slice_normalized, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    # 4. Add channel dim and convert to Tensor
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0)
    return img_tensor


def preprocess_npy_slice(slice_data, target_size):
    """Resizes a NumPy slice (already normalized [-1,1])."""
    # 1. Resize
    img_resized = resize(slice_data, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    # 2. Add channel dim and convert to Tensor
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0)
    return img_tensor


# --- Dataset ---
class SemiMAR_Dataset(Dataset):
    def __init__(self, xp_dir, low_dir, high_dir, img_size, slice_axis=2, win_min=-1000, win_max=1000):
        super().__init__()
        self.xp_dir = xp_dir
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.img_size = img_size
        self.slice_axis = slice_axis
        self.win_min = win_min
        self.win_max = win_max

        # Use Xp files to define the dataset length and find corresponding Xlow
        self.xp_files = sorted(glob.glob(os.path.join(self.xp_dir, '*_slice????_p.npy')))
        if not self.xp_files:
            raise FileNotFoundError(f"No '*_slice????_p.npy' files found in {self.xp_dir}")

        # Get list of available high (clean) NIfTI files for random sampling
        self.high_nifti_files = glob.glob(os.path.join(self.high_dir, '*.nii.gz'))
        if not self.high_nifti_files:
            raise FileNotFoundError(f"No '.nii.gz' files found in {self.high_dir}")
        # Pre-load high nifti data or paths to slices for efficiency?
        # For simplicity now, we load NIfTI file on demand, could be slow.
        # Consider pre-extracting all high slices if performance is an issue.
        print(f"Found {len(self.xp_files)} Xp slices (defines dataset size).")
        print(f"Found {len(self.high_nifti_files)} Xhigh NIfTI volumes for random sampling.")

        # Regex to parse Xp filename
        self.filename_pattern = re.compile(r"^(.*)_slice(\d{4})_p\.npy$")

    def __len__(self):
        return len(self.xp_files)

    def __getitem__(self, idx):
        xp_filepath = self.xp_files[idx]
        base_xp_filename = os.path.basename(xp_filepath)

        match = self.filename_pattern.match(base_xp_filename)
        if not match:
            print(f"Warning: Could not parse filename {base_xp_filename}. Skipping index {idx}.")
            # Return item from index 0 as fallback (or handle differently)
            return self.__getitem__(0)

        base_filename_low = match.group(1)  # e.g., dataset7_CLINIC_metal_0000_data
        slice_index = int(match.group(2))

        # Construct path to original Xlow NIfTI file
        original_low_nii_filename = f"{base_filename_low}.nii.gz"
        original_low_nii_filepath = os.path.join(self.low_dir, original_low_nii_filename)

        try:
            # --- Load and preprocess Xp ---
            xp_slice_data = np.load(xp_filepath)  # Already [-1, 1]
            xp_tensor = preprocess_npy_slice(xp_slice_data, self.img_size)

            # --- Load and preprocess Xlow ---
            low_nii = nib.load(original_low_nii_filepath)
            low_data = low_nii.get_fdata()
            if self.slice_axis == 0:
                low_slice_data = low_data[slice_index, :, :]
            elif self.slice_axis == 1:
                low_slice_data = low_data[:, slice_index, :]
            else:
                low_slice_data = low_data[:, :, slice_index]
            low_tensor = preprocess_nifti_slice(low_slice_data, self.img_size, self.win_min, self.win_max)

            # --- Load and preprocess random Xhigh slice ---
            random_high_nii_path = random.choice(self.high_nifti_files)
            high_nii = nib.load(random_high_nii_path)
            high_data = high_nii.get_fdata()
            # Select a random slice index from this volume
            num_high_slices = high_data.shape[self.slice_axis]
            random_high_slice_idx = random.randint(0, num_high_slices - 1)
            if self.slice_axis == 0:
                high_slice_data = high_data[random_high_slice_idx, :, :]
            elif self.slice_axis == 1:
                high_slice_data = high_data[:, random_high_slice_idx, :]
            else:
                high_slice_data = high_data[:, :, random_high_slice_idx]
            high_tensor = preprocess_nifti_slice(high_slice_data, self.img_size, self.win_min, self.win_max)

        except Exception as e:
            print(f"Error loading/processing data for index {idx} (Xp: {base_xp_filename}, Slice: {slice_index}): {e}")
            # Return item from index 0 as fallback
            return self.__getitem__(0)

        return {'low': low_tensor, 'prior': xp_tensor, 'high': high_tensor}


# --- Loss Functions (Keep as before) ---
criterion_GAN = nn.MSELoss()  # LSGAN loss often works well
criterion_Cycle = nn.L1Loss()
criterion_Artifact = nn.L1Loss()
criterion_ModelError = nn.L1Loss()
criterion_ImagePrior = nn.L1Loss()
criterion_Contrastive = nn.L1Loss()  # Implement carefully based on paper


def contrastive_loss(feat_anchor, feat_positive, feat_negative, epsilon=1e-6):
    """Calculates the contrastive loss as defined in the paper (Eq. 14)."""
    dist_pos = torch.mean(torch.abs(feat_positive - feat_anchor))  # Mean L1 distance
    dist_neg = torch.mean(torch.abs(feat_negative - feat_anchor))
    loss = dist_pos / (dist_neg + epsilon)  # Add epsilon for numerical stability
    return loss


# --- Initialization ---
# TODO: Determine correct ngf_dec_in based on Encoder output channels
# If Encoder uses ngf=32, downsamples twice (x2, x2), output is 128 channels.
netG = Generator(ngf_enc=32, ngf_dec_in=128, n_blocks=4).to(DEVICE)
netD_l = Discriminator().to(DEVICE)
netD_h = Discriminator().to(DEVICE)

# Optimizers
optimizer_G = optim.Adam(netG.parameters(), lr=LR_G, betas=BETAS)
optimizer_D_l = optim.Adam(netD_l.parameters(), lr=LR_D, betas=BETAS)
optimizer_D_h = optim.Adam(netD_h.parameters(), lr=LR_D, betas=BETAS)

# Learning rate schedulers
lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=LR_DECAY_EPOCH, gamma=0.5)
lr_scheduler_D_l = optim.lr_scheduler.StepLR(optimizer_D_l, step_size=LR_DECAY_EPOCH, gamma=0.5)
lr_scheduler_D_h = optim.lr_scheduler.StepLR(optimizer_D_h, step_size=LR_DECAY_EPOCH, gamma=0.5)

# Dataset and DataLoader
print("Initializing Dataset...")
try:
    dataset = SemiMAR_Dataset(
        xp_dir=DATA_DIR_XP,
        low_dir=DATA_DIR_LOW,
        high_dir=DATA_DIR_HIGH,
        img_size=IMG_SIZE,
        slice_axis=SLICE_AXIS,
        win_min=WINDOW_MIN,
        win_max=WINDOW_MAX
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    print("Dataset Initialized Successfully.")
except FileNotFoundError as e:
    print(f"Error initializing dataset: {e}")
    exit()

# --- Training Loop (Keep as before) ---
print("Starting Training Loop...")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

for epoch in range(EPOCHS):
    netG.train()
    netD_l.train()
    netD_h.train()
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        # Get data
        X_low = batch['low'].to(DEVICE)  # [-1, 1]
        X_p = batch['prior'].to(DEVICE)  # [-1, 1] (Pseudo labels from ACDNet)
        X_high = batch['high'].to(DEVICE)  # [-1, 1] (Unpaired clean)

        # --- Train Generator ---
        optimizer_G.zero_grad()

        # Generate images
        X_lh, feat_lh = netG(X_low)  # Corrected image + features
        X_hh, feat_hh = netG(X_high)  # Reconstructed clean image + features

        # Calculate artifacts (Note: X_low and X_lh are in [-1, 1])
        # Artifact = Corrupted - Clean. Need to be careful with range.
        # Paper assumes additive artifacts in original HU space.
        # Let's calculate difference in normalized space for simplicity, might need adjustment.
        X_art1 = X_low - X_lh

        # Synthesize corrupted image and reconstruct
        X_hl = torch.clamp(X_high + X_art1.detach(), -1.0, 1.0)  # Add artifact and clamp
        X_hlh, _ = netG(X_hl)  # Reconstructed clean from synthetic corrupted
        X_art2 = X_hl - X_hlh

        # Cycle reconstruction
        # X_lhl = torch.clamp(X_lh + X_art2.detach(), -1.0, 1.0) # Add artifact back and clamp
        # Using X_low for cycle consistency might be more stable
        X_lhl, _ = netG(X_low)  # Re-run X_low -> X_lh

        # --- Generator Losses ---
        pred_fake_l = netD_l(X_hl)
        loss_G_adv_l = criterion_GAN(pred_fake_l, torch.ones_like(pred_fake_l))
        pred_fake_h = netD_h(X_lh)
        loss_G_adv_h = criterion_GAN(pred_fake_h, torch.ones_like(pred_fake_h))
        loss_G_adv = loss_G_adv_l + loss_G_adv_h

        loss_cycle_l = criterion_Cycle(X_lhl, X_low)  # How well X_low is reconstructed after cycle
        loss_cycle_h = criterion_Cycle(X_hlh, X_high)  # How well X_high is reconstructed
        loss_G_cyc = loss_cycle_l + loss_cycle_h

        loss_G_art = criterion_Artifact(X_art1, X_art2)
        loss_G_model_error = criterion_ModelError(X_hh, X_high)  # Identity loss
        loss_G_im = criterion_ImagePrior(X_lh, X_p)  # Supervised loss from pseudo-label

        # Contrastive Loss
        with torch.no_grad():
            _, feat_p = netG(X_p)  # Features of pseudo-label
            # feat_hh contains features for X_high
        loss_G_cont = contrastive_loss(feat_lh, feat_p.detach(), feat_hh.detach())

        # Total Generator Loss
        loss_G = (loss_G_adv +
                  lambda_cyc * loss_G_cyc +
                  lambda_art * loss_G_art +
                  lambda_model_error * loss_G_model_error +
                  lambda_im * loss_G_im +
                  lambda_cont * loss_G_cont)

        loss_G.backward()
        optimizer_G.step()

        # --- Train Discriminator D_l ---
        optimizer_D_l.zero_grad()
        pred_real_l = netD_l(X_low)
        loss_D_real_l = criterion_GAN(pred_real_l, torch.ones_like(pred_real_l))
        pred_fake_l = netD_l(X_hl.detach())
        loss_D_fake_l = criterion_GAN(pred_fake_l, torch.zeros_like(pred_fake_l))
        loss_D_l = (loss_D_real_l + loss_D_fake_l) * 0.5
        loss_D_l.backward()
        optimizer_D_l.step()

        # --- Train Discriminator D_h ---
        optimizer_D_h.zero_grad()
        pred_real_h = netD_h(X_high)
        loss_D_real_h = criterion_GAN(pred_real_h, torch.ones_like(pred_real_h))
        pred_fake_h = netD_h(X_lh.detach())
        loss_D_fake_h = criterion_GAN(pred_fake_h, torch.zeros_like(pred_fake_h))
        loss_D_h = (loss_D_real_h + loss_D_fake_h) * 0.5
        loss_D_h.backward()
        optimizer_D_h.step()

        epoch_loss_g += loss_G.item()
        epoch_loss_d += (loss_D_l + loss_D_h).item()
        num_batches += 1

        # --- Logging ---
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch + 1}/{EPOCHS}] [Batch {i + 1}/{len(dataloader)}] "
                  f"[D loss: {(loss_D_l + loss_D_h).item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}]")

    # --- End of Epoch ---
    avg_loss_g = epoch_loss_g / num_batches
    avg_loss_d = epoch_loss_d / num_batches
    print(f"--- Epoch {epoch + 1}/{EPOCHS} Summary ---")
    print(f"Avg G Loss: {avg_loss_g:.4f} | Avg D Loss: {avg_loss_d:.4f}")

    lr_scheduler_G.step()
    lr_scheduler_D_l.step()
    lr_scheduler_D_h.step()

    # Save checkpoints
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, f'netG_epoch_{epoch + 1}.pth'))
        torch.save(netD_l.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_l_epoch_{epoch + 1}.pth'))
        torch.save(netD_h.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_h_epoch_{epoch + 1}.pth'))
        print(f"Saved checkpoint for epoch {epoch + 1}")

print("Training Finished.")

