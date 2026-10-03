# Water Segmentation using Multispectral Data

## Project Overview

This project focuses on binary water segmentation from multispectral and remote-sensing imagery using a U-Net model implemented from scratch in PyTorch.

The dataset contains 12 input channels for each image, including multispectral bands and additional remote-sensing information. The goal is to classify each pixel as either water or non-water.

## Dataset

- Number of paired images: 306
- Image size: 128 × 128 pixels
- Input channels: 12
- Label format: Binary mask
- Class 0: Non-water
- Class 1: Water
- Ground sampling distance (GSD): 30 m

The dataset was split into:

- Training: 214 images
- Validation: 46 images
- Test: 46 images

The split was stratified according to whether an image contains water.

## Input Data

The 12 input channels are:

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

The data were normalized using training-set statistics separately for each channel.

Missing values in the MERIT DEM channel were handled during preprocessing.

## Data Preprocessing

The preprocessing pipeline includes:

- Loading the 12-channel TIFF images
- Handling invalid values
- Per-channel normalization using training-set statistics
- Converting images from `(H, W, C)` to `(C, H, W)`
- Producing tensors with shape `(12, 128, 128)`
- Converting binary masks to tensors with shape `(1, 128, 128)`

Training augmentation includes spatial transformations such as horizontal flips, vertical flips, and 90-degree rotations.

## Model

A U-Net architecture was implemented from scratch in PyTorch.

The model uses:

- 12 input channels
- Encoder-decoder architecture
- Skip connections
- Batch normalization
- ReLU activations
- A single output channel for binary segmentation
- No pretrained weights

The baseline model contains approximately 31 million trainable parameters.

## Evaluation Metrics

The model was evaluated using metrics focused on the water class:

- Intersection over Union (IoU)
- Precision
- Recall
- F1-score

## Experiments

Several experiments were performed:

### 1. Baseline U-Net

The baseline uses the original 12 input channels with BCE loss.

### 2. MNDWI Feature

MNDWI was calculated from the Green and SWIR1 bands and added as an additional input feature.

This experiment improved validation performance slightly but did not improve performance on the held-out test set.

### 3. Dice + BCE Loss

A combined Dice and Binary Cross-Entropy loss was tested to improve segmentation performance.

The resulting test performance was very close to the baseline.

### 4. Weighted BCE + Dice

A weighted BCE component was combined with Dice loss to increase attention to the water class.

This increased recall but also produced more false-positive predictions and resulted in lower validation IoU.

## Final Results

| Experiment | Input | Validation IoU | Test IoU | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 12 bands | 0.7133 | 0.7750 | 0.9175 | 0.8330 | 0.8732 |
| MNDWI + U-Net | 12 bands + MNDWI | 0.7177 | 0.7516 | 0.8621 | 0.8543 | 0.8582 |
| Dice + BCE | 12 bands | 0.7201 | 0.7756 | 0.9154 | 0.8355 | 0.8736 |
| Weighted BCE + Dice | 12 bands | 0.6805 | — | — | — | — |

The default prediction threshold of 0.50 was retained based on validation-set threshold analysis.

## Error Analysis

Error analysis showed that the main remaining difficulty is detecting thin, small, and low-area water regions.

These regions were frequently missed by both the baseline model and the Dice + BCE experiment.

## Conclusion

This project developed a U-Net model from scratch for binary water segmentation using 12-channel multispectral and auxiliary remote-sensing data.

The baseline model achieved a test IoU of 0.7750 and a test F1-score of 0.8732. The experiments showed that adding MNDWI as an additional feature improved validation performance but did not improve performance on the held-out test set. The Dice + BCE loss produced performance very close to the baseline, while the weighted BCE + Dice experiment increased recall but reduced validation IoU because of increased false-positive predictions.

Overall, the experiments demonstrate that the 12-channel U-Net provides a strong baseline for this dataset, while the tested feature-engineering and loss-function modifications did not provide a consistent improvement on the test set.
