import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# Assume image loading/saving utilities are available (e.g., using PIL, SimpleITK, pydicom)
# from utils import load_image, save_image, normalize, denormalize

# --- Configuration ---
IMG_SIZE = 256 # Example image size
INPUT_CHANNELS = 1 # Grayscale CT images
OUTPUT_CHANNELS = 1
LR_G = 1e-4 # Learning rate for generator
LR_D = 1e-4 # Learning rate for discriminator
BETAS = (0.5, 0.999) # Adam optimizer betas
EPOCHS = 60
LR_DECAY_EPOCH = 20
BATCH_SIZE = 4 # Adjust based on GPU memory
DATA_DIR_LOW = 'path/to/X_low/' # Metal-corrupted images
DATA_DIR_PRIOR = 'path/to/X_p/' # Corresponding pseudo-labels
DATA_DIR_HIGH = 'path/to/X_high/' # Unpaired clean images
CHECKPOINT_DIR = './checkpoints'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights (adjust based on paper/experiments)
lambda_cyc = 20.0
lambda_art = 20.0
lambda_model_error = 20.0
lambda_im = 10.0 # For simulated data; 1.0 for clinical
lambda_cont = 1.0

# --- Model Architecture ---

# Residual Block (Helper)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Use ReflectionPad for less border artifacts
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

# Encoder (NetE) - Based on SemiMAR.pdf structure
class EncoderNet(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, ngf=32, n_blocks=4): # ngf is base number of filters
        super(EncoderNet, self).__init__()
        model = [
            nn.ReflectionPad2d(3), # Pad = (kernel_size - 1) / 2 = (7-1)/2 = 3
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers
        in_features = ngf
        out_features = ngf * 2
        # C64K4S2P1
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        out_features = in_features * 2
        # C128K4S2P1
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        # Residual blocks (e.g., 4 as shown in SemiMAR.pdf)
        # C128K3S1P1 inside blocks
        self.res_blocks_start_dim = out_features
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks += [ResidualBlock(out_features)]
        self.res_blocks_seq = nn.Sequential(*res_blocks)

        self.main_seq = nn.Sequential(*model)


    def forward(self, input):
        # Pass through initial convs and downsampling
        down_features = self.main_seq(input)
        # Pass through residual blocks
        res_output = self.res_blocks_seq(down_features)
        # Return the output of the last residual block for contrastive loss
        return res_output

# Decoder (NetD) - Based on SemiMAR.pdf structure
class DecoderNet(nn.Module):
    def __init__(self, output_nc=OUTPUT_CHANNELS, ngf=128, n_blocks=4): # ngf here matches output of encoder res blocks
        super(DecoderNet, self).__init__()

        # Residual blocks (matching encoder)
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks += [ResidualBlock(ngf)] # C128K3S1P1 inside blocks
        model = res_blocks

        # Upsampling Layers
        in_features = ngf
        out_features = in_features // 2
        # C128K5S1P2 -> Transposed Conv equivalent: k=4, s=2, p=1 or k=3, s=2, p=1, op=1?
        # Let's assume standard upsampling blocks (TransposedConv)
        # Upsample 1 (to match second downsample output dim: 64)
        model += [
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        out_features = in_features // 2
         # Upsample 2 (to match first downsample output dim: 32)
        model += [
             nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        # Output layer (CIK7S1P3)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, output_nc, kernel_size=7, padding=0),
            nn.Tanh() # Often used to map output to [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Generator combining Encoder and Decoder
class Generator(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, output_nc=OUTPUT_CHANNELS, ngf_enc=32, ngf_dec_in=128, n_blocks=4):
        super(Generator, self).__init__()
        self.encoder = EncoderNet(input_nc, ngf_enc, n_blocks)
        # ngf_dec_in should match the output channels of the last res block in encoder
        self.decoder = DecoderNet(output_nc, ngf_dec_in, n_blocks)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output, features # Return features for contrastive loss

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, input_nc=INPUT_CHANNELS, ndf=64, n_layers=3): # ndf: base number of filters
        super(Discriminator, self).__init__()
        kw = 4 # Kernel size
        padw = 1 # Padding
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers): # Gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult), # Use InstanceNorm
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # Output a 1-channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input) # Output is a patch map

# --- Dataset ---
class SemiMAR_Dataset(Dataset):
    def __init__(self, low_dir, prior_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.prior_dir = prior_dir
        self.high_dir = high_dir
        self.low_files = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)])
        self.prior_files = sorted([os.path.join(prior_dir, f) for f in os.listdir(prior_dir)])
        self.high_files = [os.path.join(high_dir, f) for f in os.listdir(high_dir)]
        self.transform = transform
        # Ensure low and prior files match
        assert len(self.low_files) == len(self.prior_files), "Mismatch between low-res and prior files count"
        # Basic check for filename correspondence could be added here

    def __len__(self):
        return len(self.low_files) # Length based on paired data

    def __getitem__(self, idx):
        low_path = self.low_files[idx]
        prior_path = self.prior_files[idx]
        # Randomly sample an unpaired high-res image
        high_idx = np.random.randint(0, len(self.high_files))
        high_path = self.high_files[high_idx]

        # --- Load images ---
        # Replace with your actual image loading logic (pydicom, SimpleITK, etc.)
        # Example using placeholder function 'load_image'
        img_low = load_image(low_path) # Should return a NumPy array [H, W] or [H, W, C]
        img_prior = load_image(prior_path)
        img_high = load_image(high_path)

        # --- Apply transformations (resizing, normalization, ToTensor) ---
        # Example: Convert to tensor, normalize to [-1, 1]
        # img_low = torch.tensor(normalize(img_low)).float().unsqueeze(0) # Add channel dim
        # img_prior = torch.tensor(normalize(img_prior)).float().unsqueeze(0)
        # img_high = torch.tensor(normalize(img_high)).float().unsqueeze(0)

        # Placeholder for actual loading and transform
        img_low = torch.randn(INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
        img_prior = torch.randn(INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
        img_high = torch.randn(INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)

        if self.transform:
             # Apply any additional transforms if needed (e.g., data augmentation)
             pass # Apply self.transform here

        return {'low': img_low, 'prior': img_prior, 'high': img_high}

# --- Loss Functions ---
criterion_GAN = nn.MSELoss() # LSGAN loss often works well
criterion_Cycle = nn.L1Loss()
criterion_Artifact = nn.L1Loss()
criterion_ModelError = nn.L1Loss()
criterion_ImagePrior = nn.L1Loss()
criterion_Contrastive = nn.L1Loss() # The paper uses a ratio, implement carefully

def contrastive_loss(feat_anchor, feat_positive, feat_negative, epsilon=1e-6):
    """Calculates the contrastive loss as defined in the paper (Eq. 14)."""
    # Using L1 distance as per the formula || . ||_1
    dist_pos = torch.mean(torch.abs(feat_positive - feat_anchor)) # Mean L1 distance
    dist_neg = torch.mean(torch.abs(feat_negative - feat_anchor))
    # Loss = distance_positive / distance_negative
    loss = dist_pos / (dist_neg + epsilon) # Add epsilon for numerical stability
    return loss

# --- Initialization ---
# Generator (NetE + NetD)
netG = Generator().to(DEVICE)
# Discriminators
netD_l = Discriminator().to(DEVICE) # Discriminator for low-res/corrupted domain
netD_h = Discriminator().to(DEVICE) # Discriminator for high-res/clean domain

# Optimizers
optimizer_G = optim.Adam(netG.parameters(), lr=LR_G, betas=BETAS)
optimizer_D_l = optim.Adam(netD_l.parameters(), lr=LR_D, betas=BETAS)
optimizer_D_h = optim.Adam(netD_h.parameters(), lr=LR_D, betas=BETAS)

# Learning rate schedulers
lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=LR_DECAY_EPOCH, gamma=0.5)
lr_scheduler_D_l = optim.lr_scheduler.StepLR(optimizer_D_l, step_size=LR_DECAY_EPOCH, gamma=0.5)
lr_scheduler_D_h = optim.lr_scheduler.StepLR(optimizer_D_h, step_size=LR_DECAY_EPOCH, gamma=0.5)

# Dataset and DataLoader
# Add appropriate transforms (resizing, normalization) here
dataset = SemiMAR_Dataset(DATA_DIR_LOW, DATA_DIR_PRIOR, DATA_DIR_HIGH, transform=None)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Training Loop ---
print("Starting Training Loop...")
for epoch in range(EPOCHS):
    netG.train()
    netD_l.train()
    netD_h.train()
    for i, batch in enumerate(dataloader):
        # Get data
        X_low = batch['low'].to(DEVICE)
        X_p = batch['prior'].to(DEVICE) # Pseudo labels
        X_high = batch['high'].to(DEVICE)

        # --- Train Generator ---
        optimizer_G.zero_grad()

        # Generate images
        X_lh, feat_lh = netG(X_low) # Corrected image + features
        X_hh, feat_hh = netG(X_high) # Reconstructed clean image + features (for model error loss)

        # Calculate artifacts
        X_art1 = X_low - X_lh

        # Synthesize corrupted image and reconstruct
        X_hl = X_high + X_art1.detach() # Detach art1 to avoid grads flowing back through it here
        X_hlh, _ = netG(X_hl) # Reconstructed clean from synthetic corrupted
        X_art2 = X_hl - X_hlh

        # Cycle reconstruction
        X_lhl, _ = netG(X_low) # Re-run X_low to get X_lh for cycle consistency
        X_lhl = X_lh + X_art2.detach() # Use detached art2

        # --- Generator Losses ---
        # Adversarial Loss (Generator wants discriminators to think generated images are real)
        pred_fake_l = netD_l(X_hl) # Discriminator's prediction for synthetic corrupted
        loss_G_adv_l = criterion_GAN(pred_fake_l, torch.ones_like(pred_fake_l)) # Target is 1 (real)

        pred_fake_h = netD_h(X_lh) # Discriminator's prediction for corrected image
        loss_G_adv_h = criterion_GAN(pred_fake_h, torch.ones_like(pred_fake_h)) # Target is 1 (real)

        loss_G_adv = loss_G_adv_l + loss_G_adv_h

        # Cycle Consistency Loss
        loss_cycle_l = criterion_Cycle(X_lhl, X_low)
        loss_cycle_h = criterion_Cycle(X_hlh, X_high) # Should reconstruct clean image
        loss_G_cyc = loss_cycle_l + loss_cycle_h

        # Artifact Consistency Loss
        loss_G_art = criterion_Artifact(X_art1, X_art2)

        # Model Error Loss (Identity Loss for clean images)
        loss_G_model_error = criterion_ModelError(X_hh, X_high)

        # Image Prior Loss (Supervision from pre-trained model)
        loss_G_im = criterion_ImagePrior(X_lh, X_p)

        # Contrastive Loss (Requires features)
        # Need features for X_p and X_h as well
        with torch.no_grad(): # Don't need gradients for prior/high features here
             _, feat_p = netG(X_p)
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

        # --- Train Discriminator D_l (Corrupted Domain) ---
        optimizer_D_l.zero_grad()

        # Real loss
        pred_real_l = netD_l(X_low)
        loss_D_real_l = criterion_GAN(pred_real_l, torch.ones_like(pred_real_l))

        # Fake loss
        pred_fake_l = netD_l(X_hl.detach()) # Use detached generated image
        loss_D_fake_l = criterion_GAN(pred_fake_l, torch.zeros_like(pred_fake_l))

        # Total D_l loss
        loss_D_l = (loss_D_real_l + loss_D_fake_l) * 0.5
        loss_D_l.backward()
        optimizer_D_l.step()

        # --- Train Discriminator D_h (Clean Domain) ---
        optimizer_D_h.zero_grad()

        # Real loss
        pred_real_h = netD_h(X_high)
        loss_D_real_h = criterion_GAN(pred_real_h, torch.ones_like(pred_real_h))

        # Fake loss (using the corrected image X_lh)
        pred_fake_h = netD_h(X_lh.detach()) # Use detached generated image
        loss_D_fake_h = criterion_GAN(pred_fake_h, torch.zeros_like(pred_fake_h))

        # Total D_h loss
        loss_D_h = (loss_D_real_h + loss_D_fake_h) * 0.5
        loss_D_h.backward()
        optimizer_D_h.step()

        # --- Logging ---
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i+1}/{len(dataloader)}] "
                  f"[D loss: {(loss_D_l + loss_D_h).item():.4f}] "
                  f"[G loss: {loss_G.item():.4f} | "
                  f"adv: {loss_G_adv.item():.4f}, cyc: {loss_G_cyc.item():.4f}, art: {loss_G_art.item():.4f}, "
                  f"model: {loss_G_model_error.item():.4f}, prior: {loss_G_im.item():.4f}, cont: {loss_G_cont.item():.4f}]")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_l.step()
    lr_scheduler_D_h.step()

    # Save checkpoints (optional)
    if (epoch + 1) % 10 == 0:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, f'netG_epoch_{epoch+1}.pth'))
        torch.save(netD_l.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_l_epoch_{epoch+1}.pth'))
        torch.save(netD_h.state_dict(), os.path.join(CHECKPOINT_DIR, f'netD_h_epoch_{epoch+1}.pth'))
        print(f"Saved checkpoint for epoch {epoch+1}")

print("Training Finished.")

# --- Placeholder for Image Loading ---
# Replace this with your actual DICOM/NIfTI/PNG loading logic
def load_image(path):
    # Example: Load a numpy array, apply windowing/normalization, resize
    # img_array = np.load(path) # If saved as .npy
    # Apply windowing...
    # Apply normalization...
    # Resize...
    # Return numpy array HxW
    # For now, return random data matching expected dimensions before transform
    return np.random.rand(IMG_SIZE, IMG_SIZE)

# --- Placeholder for Normalization ---
# Replace with appropriate normalization for your data range (e.g., HU to [-1, 1])
def normalize(img_array):
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    if max_val > min_val:
        return 2 * (img_array - min_val) / (max_val - min_val) - 1
    else:
        return np.zeros_like(img_array) # Handle constant image case

