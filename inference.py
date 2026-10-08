
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tifffile


# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "water_unet_deployment.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    """Two convolutional layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """12-channel U-Net matching the trained Week 1 model."""

    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )
        self.decoder4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(128, 64)

        self.output = nn.Conv2d(
            64, out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.output(dec1)


def load_checkpoint(path):
    """Load the model checkpoint on the available device."""

    try:
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    required_keys = {
        "model_state_dict",
        "band_means",
        "band_stds",
        "input_channels",
        "image_size",
        "threshold",
    }

    missing = required_keys.difference(checkpoint.keys())
    if missing:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing)}"
        )

    if checkpoint["input_channels"] != 12:
        raise ValueError("The checkpoint must contain a 12-channel model.")

    if checkpoint["image_size"] != 128:
        raise ValueError("The expected image size is 128 x 128.")

    means = np.asarray(checkpoint["band_means"], dtype=np.float32)
    stds = np.asarray(checkpoint["band_stds"], dtype=np.float32)

    if means.shape != (12,) or stds.shape != (12,):
        raise ValueError("Expected 12 normalization means and standard deviations.")

    if not np.all(np.isfinite(means)):
        raise ValueError("Normalization means contain invalid values.")

    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError("Normalization standard deviations must be positive.")

    network = UNet(in_channels=12, out_channels=1)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(DEVICE)
    network.eval()

    return network, means, stds, float(checkpoint["threshold"])


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}"
    )

model, BAND_MEANS, BAND_STDS, THRESHOLD = load_checkpoint(MODEL_PATH)


def preprocess_image(image):
    """Convert a raw 12-channel image into a normalized model tensor."""

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError("Expected a three-dimensional 12-channel image.")

    if image.shape[-1] == 12:
        pass
    elif image.shape[0] == 12:
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(
            f"Expected 12 channels, but received shape {image.shape}."
        )

    if image.shape != (128, 128, 12):
        raise ValueError(
            f"Expected image shape (128, 128, 12), got {image.shape}."
        )

    image = image.astype(np.float32, copy=True)

    # Replace invalid values in the MERIT DEM channel using its training mean.
    for channel in range(12):
        band = image[:, :, channel]
        invalid = ~np.isfinite(band)

        if channel == 8:
            invalid |= band == -9999

        band[invalid] = BAND_MEANS[channel]
        image[:, :, channel] = band

    # Apply the same training-set per-band standardization.
    image = (image - BAND_MEANS.reshape(1, 1, 12)) / (
        BAND_STDS.reshape(1, 1, 12)
    )

    # Convert HWC to CHW.
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image, dtype=np.float32)

    return torch.from_numpy(image).unsqueeze(0)


def predict_array(image):
    """Return a binary water mask with values 0 and 1."""

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= THRESHOLD).to(torch.uint8)

    return mask[0, 0].cpu().numpy()


def predict_tiff(file_path):
    """Load a multispectral TIFF and return its binary water mask."""

    image = tifffile.imread(str(file_path))
    return predict_array(image)
