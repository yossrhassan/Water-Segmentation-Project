# Water Segmentation using Multispectral and Optical Data

## Project Overview

This project focuses on binary water segmentation using 12-channel multispectral and auxiliary remote-sensing data.

The objective is to develop a deep learning system that takes multispectral image patches as input and predicts a binary mask identifying water and non-water pixels.

The project covers three main components:

- **Week 1:** Dataset exploration, preprocessing, and U-Net implementation from scratch.
- **Week 2:** Pretrained ResNet34 encoder with a U-Net decoder.
- **Deployment:** Saved model weights, inference pipeline, Flask REST API, and HTML frontend.

## Dataset

The dataset contains 306 paired satellite images and binary water masks.

- Number of paired samples: 306
- Image size: 128 × 128 pixels
- Input channels: 12
- Label format: Binary mask
- Class 0: Non-water
- Class 1: Water
- Ground sampling distance (GSD): 30 m

The dataset was split into training, validation, and test sets:

| Split | Images |
|---|---:|
| Training | 214 |
| Validation | 46 |
| Test | 46 |
| Total | 306 |

The split was stratified according to whether each image contains water. The same fixed split was used for the experiments to support a fair comparison.

The dataset is stored separately and is not included in the GitHub repository.

## Input Data

The 12 input channels represent multispectral and auxiliary remote-sensing information:

1. Coastal Aerosol
2. Blue
3. Green
4. Red
5. Near Infrared (NIR)
6. Short-Wave Infrared 1 (SWIR1)
7. Short-Wave Infrared 2 (SWIR2)
8. QA Band
9. MERIT DEM
10. Copernicus DEM
11. ESA WorldCover Map
12. Water Occurrence Probability

The first seven channels provide spectral information, while the remaining channels provide quality, elevation, land-cover, and water-occurrence information.

## Preprocessing

The preprocessing pipeline includes:

- Loading multispectral TIFF images.
- Matching image files with their corresponding binary masks.
- Handling invalid values in the MERIT DEM channel.
- Normalizing each band separately using training-set statistics.
- Converting images from HWC format to CHW format.
- Producing input tensors with shape `(12, 128, 128)`.
- Producing binary mask tensors with shape `(1, 128, 128)`.

Missing or invalid MERIT DEM values represented by `-9999` are replaced using the corresponding training-set mean during preprocessing.

Training augmentation includes horizontal flips, vertical flips, and rotations by multiples of 90 degrees. Validation and test images are not augmented.

## Week 1 — Scratch U-Net

A U-Net architecture was implemented from scratch using PyTorch without pretrained weights.

The baseline model includes:

- 12 input channels
- Encoder-decoder architecture
- Skip connections
- Batch normalization
- ReLU activations
- A single output channel for binary segmentation

The model contains approximately 31 million trainable parameters.

### Week 1 Experiments

Several configurations were evaluated:

- Baseline U-Net with Binary Cross-Entropy loss.
- U-Net with an additional MNDWI feature.
- U-Net with Dice + BCE loss.
- U-Net with weighted BCE + Dice loss.

The baseline U-Net was retained as the principal reference model.

### Week 1 Results

| Experiment | Input | Validation IoU | Test IoU | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 12 bands | 0.7133 | 0.7750 | 0.9175 | 0.8330 | 0.8732 |
| MNDWI + U-Net | 12 bands + MNDWI | 0.7177 | 0.7516 | 0.8621 | 0.8543 | 0.8582 |
| Dice + BCE | 12 bands | 0.7201 | 0.7756 | 0.9154 | 0.8355 | 0.8736 |
| Weighted BCE + Dice | 12 bands | 0.6805 | — | — | — | — |

The weighted BCE + Dice experiment was not evaluated on the test set because its validation IoU was lower than that of the baseline.

The MNDWI experiment improved validation IoU slightly but did not improve test-set performance. The Dice + BCE experiment produced results very close to the baseline.

## Week 2 — Pretrained Segmentation Model

Week 2 introduced transfer learning using the `segmentation-models-pytorch` library.

A U-Net with a pretrained ResNet34 encoder was selected for fine-tuning on the 12-channel water segmentation dataset.

### Adapting the Encoder to 12 Channels

The original ResNet34 first convolution is designed for three-channel RGB input.

To support the multispectral data, the first convolution was adapted to accept 12 input channels.

- The original pretrained weights for the first three channels were preserved.
- The additional nine input channels were initialized using the mean of the pretrained RGB convolution weights.

The resulting first convolution accepts 12 input channels and produces 64 output channels.

### Fine-tuning Configuration

- Architecture: ResNet34 encoder + U-Net decoder
- Input channels: 12
- Output channels: 1
- Encoder weights: ImageNet pretrained
- Loss function: Binary Cross-Entropy with Logits
- Optimizer: Adam
- Learning rate: `1e-4`
- Maximum epochs: 20
- Model selection: Best validation IoU
- Prediction threshold: 0.50

### Week 2 Results

| Model | Validation IoU | Test IoU | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| Scratch U-Net | 0.7133 | 0.7750 | 0.9175 | 0.8330 | 0.8732 |
| Pretrained ResNet34 + U-Net | 0.7257 | 0.7330 | 0.8656 | 0.8271 | 0.8459 |

The pretrained model achieved a higher validation IoU than the Week 1 scratch U-Net:

**0.7257 compared with 0.7133.**

However, the scratch U-Net achieved higher IoU and F1-score on the held-out test set. This indicates that the validation improvement did not translate into better test-set performance in this experiment.

## Error Analysis

Visual comparisons showed that both models can identify the main water structures in some test images.

However, both models struggle with:

- Thin water channels.
- Small water bodies.
- Low-area water regions.
- Difficult or low-contrast water structures.

The pretrained model did not demonstrate a consistent visual advantage over the scratch U-Net.

## Overall Model Conclusion

The experiments show that the scratch U-Net is a strong baseline for this dataset. The pretrained ResNet34 + U-Net achieved a measurable improvement in validation IoU, but did not outperform the scratch model on the held-out test set.

The results demonstrate that ImageNet pretraining can provide a useful initialization, but adapting an RGB-pretrained encoder to multispectral and auxiliary remote-sensing data does not necessarily guarantee better generalization.

## Deployment — Flask REST API

A local Flask application was implemented to serve the trained U-Net model through a REST API and a simple HTML frontend.

The application accepts a 12-channel multispectral TIFF image, performs preprocessing and inference, and returns a binary water segmentation mask as a PNG image.

### Project Files

- `Water_Segmentation_Multispectral_UNet.ipynb`: Original Week 1 notebook.
- `Water_Segmentation_Multispectral_UNet_Final.ipynb`: Consolidated notebook containing the project experiments.
- `app.py`: Flask application, API endpoints, and HTML frontend.
- `inference.py`: U-Net architecture, checkpoint loading, preprocessing, and prediction functions.
- `requirements.txt`: Python dependencies.
- `water_unet_deployment.pth`: Trained model checkpoint, available in the GitHub Releases section.

The sample TIFF images used during local testing are not required to be included in the repository.

### Requirements

- Python 3.14.8 or a compatible Python installation.
- The trained model checkpoint.
- The packages listed in `requirements.txt`.

### Setup

1. Download or clone the GitHub repository.
2. Download `water_unet_deployment.pth` from the repository's Releases section.
3. Place `water_unet_deployment.pth` in the project root, alongside `app.py` and `inference.py`.
4. Open a terminal in the project directory.
5. Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the Application

Start the Flask server:

```bash
python app.py
```

Open the application in a web browser:

http://127.0.0.1:5000

The HTML frontend allows users to select a TIFF image and view the predicted water mask.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Displays the HTML frontend. |
| `/health` | GET | Returns the API health status. |
| `/predict` | POST | Accepts a TIFF upload and returns a PNG segmentation mask. |

The `/predict` endpoint expects a multipart form upload using the field name `image`.

Supported input:

- File format: TIFF (`.tif` or `.tiff`)
- Image dimensions: 128 × 128 pixels
- Input channels: 12

The API returns a PNG image with pixel values of 0 and 255, representing non-water and predicted water, respectively.

### Local API Testing

The Flask application was tested locally through the HTML frontend using three sample TIFF images.

| Test | Input | Result |
|---|---|---|
| 1 | `0.tif` | The API returned a binary mask. |
| 2 | `1.tif` | The API returned a binary mask. |
| 3 | `2.tif` | The API returned a binary mask. |

These tests confirm that the application accepted the three sample inputs and returned mask images successfully.

Successful API responses do not, by themselves, establish the segmentation accuracy of each individual prediction. Prediction quality should be assessed separately against the corresponding ground-truth masks.

### Deployment Checkpoint Evaluation

The specific checkpoint exported for deployment was reloaded and evaluated locally on the test set.

| Metric | Result |
|---|---:|
| Test IoU | 0.7726 |
| Test Precision | 0.9381 |
| Test Recall | 0.8141 |
| Test F1-score | 0.8717 |

These metrics describe the exported deployment checkpoint and are reported separately from the original Week 1 model comparison.

The checkpoint contains the model weights and preprocessing configuration, including the per-band normalization means and standard deviations, input channel count, image size, and prediction threshold.

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385
- Chen, L.-C., et al. (2016). *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.* https://arxiv.org/abs/1606.00915
- Segmentation Models PyTorch documentation: https://smp.readthedocs.io/
- Segmentation Models PyTorch GitHub: https://github.com/qubvel-org/segmentation_models.pytorch
- Flask documentation: https://flask.palletsprojects.com/
- PyTorch documentation: https://pytorch.org/docs/
