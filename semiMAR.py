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
from skimage.transform import resize # Or use cv2
import time # <--- Import time
from datetime import timedelta # <--- Import timedelta for formatting time

# --- Configuration ---
IMG_SIZE = 416 # Example image size. (ACDNet used 416x416. 512x512 is also an option
INPUT_CHANNELS = 1 # Grayscale CT images
OUTPUT_CHANNELS = 1
LR_G = 1e-4 # Learning rate for generator (SemiMAR paper: 0.0001)
LR_D = 1e-4 # Learning rate for discriminator (SemiMAR paper: 0.0001)
BETAS = (0.5, 0.999) # Adam optimizer betas (SemiMAR paper uses these)
EPOCHS = 60 # SemiMAR paper uses 60
LR_DECAY_EPOCH = 20 # SemiMAR paper halves every 20 epochs
BATCH_SIZE = 8 # User specified
NUM_WORKERS = 14 # User specified

# --- Data Paths (User Specified) ---
DATA_DIR_XP = '/home/mcosta/PycharmProjects/Semi_ACDNet/clinic_metal_results_copy_mode/' # Directory with Xp .npy files
DATA_DIR_LOW = 'data/train/CTPelvic1K_METAL/'  # Directory with corresponding Xlow .nii.gz files
DATA_DIR_HIGH = 'data/train/CTPelvic1K_METAL_FREE/'  # Directory with unpaired Xhigh .nii.gz files
# --- End Data Paths ---
CHECKPOINT_DIR = './checkpoints_semimar'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights (from SemiMAR paper - clinical)
lambda_cyc = 20.0
lambda_art = 20.0
lambda_model_error = 20.0
lambda_im = 1.0
lambda_cont = 1.0

# Preprocessing Params
SLICE_AXIS = 2 # Axis for slicing NIfTI files
WINDOW_MIN = -1000 # HU window min for X_low/X_high
WINDOW_MAX = 1000 # HU window max for X_low/X_high

# --- Model Architecture ---

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

# Encoder (NetE)
class EncoderNet(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, ngf=32, n_blocks=4):
        super(EncoderNet, self).__init__()
        model = [
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True) ]
        in_features = ngf; out_features = ngf * 2
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        in_features = out_features; out_features = in_features * 2
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        self.initial_layers = nn.Sequential(*model)
        res_blocks = []
        for _ in range(n_blocks): res_blocks += [ResidualBlock(out_features)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)
    def forward(self, input):
        initial_out = self.initial_layers(input); res_out = self.res_blocks_seq(initial_out); return res_out

# Decoder (NetD)
class DecoderNet(nn.Module):
    def __init__(self, output_nc=OUTPUT_CHANNELS, ngf=128, n_blocks=4):
        super(DecoderNet, self).__init__()
        res_blocks = [];
        for _ in range(n_blocks): res_blocks += [ResidualBlock(ngf)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)
        model = []
        in_features = ngf; out_features = in_features // 2
        model += [
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        in_features = out_features; out_features = in_features // 2
        model += [
             nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        model += [
            nn.ReflectionPad2d(3), nn.Conv2d(out_features, output_nc, kernel_size=7, padding=0), nn.Tanh() ]
        self.upsample_layers = nn.Sequential(*model)
    def forward(self, input):
        res_out = self.res_blocks_seq(input); output = self.upsample_layers(res_out); return output

# Generator
class Generator(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, output_nc=OUTPUT_CHANNELS, ngf_enc=32, ngf_dec_in=128, n_blocks=4):
        super(Generator, self).__init__()
        self.encoder = EncoderNet(input_nc, ngf_enc, n_blocks)
        self.decoder = DecoderNet(output_nc, ngf_dec_in, n_blocks)
    def forward(self, x):
        features = self.encoder(x); output = self.decoder(features); return output, features

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        kw = 4; padw = 1; sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult; nf_mult = min(2 ** n, 8)
            sequence += [ nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, inplace=True) ]
        nf_mult_prev = nf_mult; nf_mult = min(2 ** n_layers, 8)
        sequence += [ nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult), nn.LeakyReLU(0.2, inplace=True) ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)
    def forward(self, input): return self.model(input)


# --- Utility Functions for Dataset ---
def preprocess_nifti_slice(slice_data, target_size, win_min, win_max):
    img_slice_windowed = np.clip(slice_data, win_min, win_max)
    if win_max > win_min: img_slice_normalized = 2.0 * (img_slice_windowed - win_min) / (win_max - win_min) - 1.0
    else: img_slice_normalized = np.zeros_like(img_slice_windowed)
    img_resized = resize(img_slice_normalized, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0); return img_tensor
def preprocess_npy_slice(slice_data, target_size):
    img_resized = resize(slice_data, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0); return img_tensor

# --- Dataset ---
class SemiMAR_Dataset(Dataset):
    def __init__(self, xp_dir, low_dir, high_dir, img_size, slice_axis=2, win_min=-1000, win_max=1000):
        super().__init__()
        self.xp_dir = xp_dir; self.low_dir = low_dir; self.high_dir = high_dir
        self.img_size = img_size; self.slice_axis = slice_axis
        self.win_min = win_min; self.win_max = win_max
        self.xp_files = sorted(glob.glob(os.path.join(self.xp_dir, '*_slice????_p.npy')))
        if not self.xp_files: raise FileNotFoundError(f"No '*_slice????_p.npy' files found in {self.xp_dir}")
        self.high_nifti_files = glob.glob(os.path.join(self.high_dir, '*.nii.gz'))
        if not self.high_nifti_files: raise FileNotFoundError(f"No '.nii.gz' files found in {self.high_dir}")
        print(f"Found {len(self.xp_files)} Xp slices (defines dataset size).")
        print(f"Found {len(self.high_nifti_files)} Xhigh NIfTI volumes for random sampling.")
        self.filename_pattern = re.compile(r"^(.*)_slice(\d{4})_p\.npy$")
    def __len__(self): return len(self.xp_files)
    def __getitem__(self, idx):
        xp_filepath = self.xp_files[idx]; base_xp_filename = os.path.basename(xp_filepath)
        match = self.filename_pattern.match(base_xp_filename)
        if not match: print(f"Warning: Could not parse filename {base_xp_filename}. Skipping index {idx}."); return self.__getitem__(random.randint(0, len(self.xp_files)-1)) # Return random instead of 0
        base_filename_low = match.group(1); slice_index = int(match.group(2))
        original_low_nii_filename = f"{base_filename_low}.nii.gz"
        original_low_nii_filepath = os.path.join(self.low_dir, original_low_nii_filename)
        try:
            xp_slice_data = np.load(xp_filepath); xp_tensor = preprocess_npy_slice(xp_slice_data, self.img_size)
            low_nii = nib.load(original_low_nii_filepath); low_data = low_nii.get_fdata()
            if self.slice_axis == 0: low_slice_data = low_data[slice_index, :, :]
            elif self.slice_axis == 1: low_slice_data = low_data[:, slice_index, :]
            else: low_slice_data = low_data[:, :, slice_index]
            low_tensor = preprocess_nifti_slice(low_slice_data, self.img_size, self.win_min, self.win_max)
            random_high_nii_path = random.choice(self.high_nifti_files)
            high_nii = nib.load(random_high_nii_path); high_data = high_nii.get_fdata()
            num_high_slices = high_data.shape[self.slice_axis]; random_high_slice_idx = random.randint(0, num_high_slices - 1)
            if self.slice_axis == 0: high_slice_data = high_data[random_high_slice_idx, :, :]
            elif self.slice_axis == 1: high_slice_data = high_data[:, random_high_slice_idx, :]
            else: high_slice_data = high_data[:, :, random_high_slice_idx]
            high_tensor = preprocess_nifti_slice(high_slice_data, self.img_size, self.win_min, self.win_max)
        except Exception as e: print(f"Error loading/processing data for index {idx} (Xp: {base_xp_filename}, Slice: {slice_index}): {e}"); return self.__getitem__(random.randint(0, len(self.xp_files)-1)) # Return random
        return {'low': low_tensor, 'prior': xp_tensor, 'high': high_tensor}

# --- Loss Functions ---
criterion_GAN = nn.MSELoss(); criterion_Cycle = nn.L1Loss(); criterion_Artifact = nn.L1Loss()
criterion_ModelError = nn.L1Loss(); criterion_ImagePrior = nn.L1Loss(); criterion_Contrastive = nn.L1Loss()
def contrastive_loss(feat_anchor, feat_positive, feat_negative, epsilon=1e-6):
    dist_pos = torch.mean(torch.abs(feat_positive - feat_anchor))
    dist_neg = torch.mean(torch.abs(feat_negative - feat_anchor))
    loss = dist_pos / (dist_neg + epsilon); return loss

# --- Initialization ---
netG = Generator(ngf_enc=32, ngf_dec_in=128, n_blocks=4).to(DEVICE)
netD_l = Discriminator().to(DEVICE); netD_h = Discriminator().to(DEVICE)
optimizer_G = optim.Adam(netG.parameters(), lr=LR_G, betas=BETAS)
optimizer_D_l = optim.Adam(netD_l.parameters(), lr=LR_D, betas=BETAS)
optimizer_D_h = optim.Adam(netD_h.parameters(), lr=LR_D, betas=BETAS)
lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=LR_DECAY_EPOCH, gamma=0.5)
lr_scheduler_D_l = optim.lr_scheduler.StepLR(optimizer_D_l, step_size=LR_DECAY_EPOCH, gamma=0.5)
lr_scheduler_D_h = optim.lr_scheduler.StepLR(optimizer_D_h, step_size=LR_DECAY_EPOCH, gamma=0.5)

# --- Dataset and DataLoader ---
print("Initializing Dataset...")
try:
    dataset = SemiMAR_Dataset(xp_dir=DATA_DIR_XP, low_dir=DATA_DIR_LOW, high_dir=DATA_DIR_HIGH,
                              img_size=IMG_SIZE, slice_axis=SLICE_AXIS, win_min=WINDOW_MIN, win_max=WINDOW_MAX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) # Use NUM_WORKERS
    print("Dataset Initialized Successfully.")
except FileNotFoundError as e: print(f"Error initializing dataset: {e}"); exit()
except Exception as e: print(f"An unexpected error occurred during dataset init: {e}"); exit() # Catch other potential errors


# --- Training Loop ---
print(f"Starting Training Loop... Device: {DEVICE}")
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

start_time_train = time.time() # <--- Record overall start time

for epoch in range(EPOCHS):
    netG.train(); netD_l.train(); netD_h.train()
    epoch_loss_g = 0.0; epoch_loss_d = 0.0
    epoch_start_time = time.time() # <--- Record epoch start time
    num_batches = 0
    batches_in_epoch = len(dataloader) # Get total number of batches

    for i, batch in enumerate(dataloader):
        batch_start_time = time.time() # <--- Optional: Record batch start time
        # --- Get data ---
        X_low = batch['low'].to(DEVICE); X_p = batch['prior'].to(DEVICE); X_high = batch['high'].to(DEVICE)

        # --- Generator Training ---
        optimizer_G.zero_grad()
        X_lh, feat_lh = netG(X_low); X_hh, feat_hh = netG(X_high)
        X_art1 = X_low - X_lh
        X_hl = torch.clamp(X_high + X_art1.detach(), -1.0, 1.0)
        X_hlh, _ = netG(X_hl); X_art2 = X_hl - X_hlh
        X_lhl, _ = netG(X_low) # Cycle
        pred_fake_l = netD_l(X_hl); loss_G_adv_l = criterion_GAN(pred_fake_l, torch.ones_like(pred_fake_l))
        pred_fake_h = netD_h(X_lh); loss_G_adv_h = criterion_GAN(pred_fake_h, torch.ones_like(pred_fake_h))
        loss_G_adv = loss_G_adv_l + loss_G_adv_h
        loss_cycle_l = criterion_Cycle(X_lhl, X_low); loss_cycle_h = criterion_Cycle(X_hlh, X_high)
        loss_G_cyc = loss_cycle_l + loss_cycle_h
        loss_G_art = criterion_Artifact(X_art1, X_art2)
        loss_G_model_error = criterion_ModelError(X_hh, X_high)
        loss_G_im = criterion_ImagePrior(X_lh, X_p)
        with torch.no_grad(): _, feat_p = netG(X_p)
        loss_G_cont = contrastive_loss(feat_lh, feat_p.detach(), feat_hh.detach())
        loss_G = (loss_G_adv + lambda_cyc * loss_G_cyc + lambda_art * loss_G_art +
                  lambda_model_error * loss_G_model_error + lambda_im * loss_G_im + lambda_cont * loss_G_cont)
        loss_G.backward(); optimizer_G.step()

        # --- Discriminator Training ---
        optimizer_D_l.zero_grad()
        pred_real_l = netD_l(X_low); loss_D_real_l = criterion_GAN(pred_real_l, torch.ones_like(pred_real_l))
        pred_fake_l = netD_l(X_hl.detach()); loss_D_fake_l = criterion_GAN(pred_fake_l, torch.zeros_like(pred_fake_l))
        loss_D_l = (loss_D_real_l + loss_D_fake_l) * 0.5; loss_D_l.backward(); optimizer_D_l.step()
        optimizer_D_h.zero_grad()
        pred_real_h = netD_h(X_high); loss_D_real_h = criterion_GAN(pred_real_h, torch.ones_like(pred_real_h))
        pred_fake_h = netD_h(X_lh.detach()); loss_D_fake_h = criterion_GAN(pred_fake_h, torch.zeros_like(pred_fake_h))
        loss_D_h = (loss_D_real_h + loss_D_fake_h) * 0.5; loss_D_h.backward(); optimizer_D_h.step()

        # --- Accumulate losses ---
        epoch_loss_g += loss_G.item(); epoch_loss_d += (loss_D_l + loss_D_h).item()
        num_batches += 1
        batch_end_time = time.time() # <--- Optional: Record batch end time

        # --- Logging (Adjust frequency if needed) ---
        log_interval = 100
        if (i + 1) % log_interval == 0 or (i + 1) == batches_in_epoch: # Log also on last batch
            batches_processed = i + 1
            time_per_batch = (batch_end_time - batch_start_time) if 'batch_start_time' in locals() else 0
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {batches_processed}/{batches_in_epoch}] "
                  f"[D loss: {(loss_D_l + loss_D_h).item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}] "
                  f"[Batch Time: {time_per_batch:.2f}s]") # <--- Optional: Print batch time

    # --- End of Epoch ---
    epoch_end_time = time.time() # <--- Record epoch end time
    epoch_duration = epoch_end_time - epoch_start_time
    total_elapsed_time = epoch_end_time - start_time_train

    avg_loss_g = epoch_loss_g / num_batches if num_batches > 0 else 0
    avg_loss_d = epoch_loss_d / num_batches if num_batches > 0 else 0

    # --- Calculate Estimated Time Remaining ---
    epochs_remaining = EPOCHS - (epoch + 1)
    # Use average epoch time only after the first epoch for better estimate
    avg_time_per_epoch = total_elapsed_time / (epoch + 1) if epoch > 0 else epoch_duration
    estimated_remaining_time = avg_time_per_epoch * epochs_remaining

    print("-" * 60) # <--- Separator
    print(f"--- Epoch {epoch+1}/{EPOCHS} Summary ---")
    print(f"Epoch Duration: {timedelta(seconds=int(epoch_duration))}") # <--- Print epoch duration
    print(f"Avg G Loss: {avg_loss_g:.4f} | Avg D Loss: {avg_loss_d:.4f}")
    print(f"Total Elapsed Time: {timedelta(seconds=int(total_elapsed_time))}") # <--- Print total elapsed time
    if epoch < EPOCHS - 1: # Don't print remaining time after last epoch
        print(f"Estimated Remaining Time: {timedelta(seconds=int(estimated_remaining_time))}") # <--- Print estimated remaining time
    print("-" * 60) # <--- Separator

    # --- LR scheduler steps and Checkpoint Saving ---
    lr_scheduler_G.step(); lr_scheduler_D_l.step(); lr_scheduler_D_h.step()
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, f'netG_epoch_{epoch+1}.pth'))
        torch.save(netD_l.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_l_epoch_{epoch+1}.pth'))
        torch.save(netD_h.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_h_epoch_{epoch+1}.pth'))
        print(f"Saved checkpoint for epoch {epoch+1}")

# --- End of Training ---
final_elapsed_time = time.time() - start_time_train
print("=" * 60)
print("Training Finished.")
print(f"Total Training Time: {timedelta(seconds=int(final_elapsed_time))}") # <--- Print final total time
print("=" * 60)

