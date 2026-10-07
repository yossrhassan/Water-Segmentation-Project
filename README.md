# Water Segmentation using Multispectral and Optical Data

## Project Overview

This project focuses on binary water segmentation using 12-channel multispectral and auxiliary remote-sensing data.

The main objective is to develop a U-Net-based segmentation system that takes 12-channel image patches as input and predicts a binary water mask at pixel level.

The project was developed in two stages:

- **Week 1:** U-Net implemented from scratch
- **Week 2:** Pretrained ResNet34 encoder with a U-Net decoder

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

The same fixed split was used for all experiments to ensure a fair comparison.

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

Invalid values in the MERIT DEM channel were handled during preprocessing.

## Week 1 — Scratch U-Net

A U-Net architecture was implemented from scratch in PyTorch.

The baseline model uses:

- 12 input channels
- Encoder-decoder architecture
- Skip connections
- Batch normalization
- ReLU activations
- One output channel for binary segmentation
- No pretrained weights

The model contains approximately 31 million trainable parameters.

### Week 1 Experiments

Several experiments were tested:

- Baseline U-Net with BCE loss
- MNDWI feature + U-Net
- Dice + BCE loss
- Weighted BCE + Dice loss

The baseline U-Net was selected as the main reference model.

## Week 2 — Pretrained Segmentation Model

Week 2 extends the Week 1 pipeline using a pretrained segmentation backbone.

A **ResNet34 encoder with a U-Net decoder** was implemented using `segmentation-models-pytorch`.

### Pretrained Encoder Adaptation

The original ResNet34 first convolution is designed for 3-channel RGB input.

To support the 12-channel dataset, the first convolution was explicitly adapted from 3 input channels to 12 input channels.

The original pretrained weights for the first three channels were preserved, while the additional nine channels were initialized using the mean of the pretrained RGB convolution weights.

The resulting first convolution accepts:

```text
12 input channels → 64 output channels
```

### Fine-tuning

The pretrained U-Net was fine-tuned on the same training set and evaluated using the same validation and test sets as the Week 1 baseline.

The optimization setup used:

- Binary Cross-Entropy with Logits loss
- Adam optimizer
- Learning rate: 1e-4
- Maximum epochs: 20
- Early stopping based on validation IoU

## Evaluation Metrics

Performance was evaluated on the water class using:

- Intersection over Union (IoU)
- Precision
- Recall
- F1-score

A prediction threshold of 0.50 was used.

## Final Comparison

| Model | Validation IoU | Test IoU | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| Scratch U-Net | 0.7133 | 0.7750 | 0.9175 | 0.8330 | 0.8732 |
| Pretrained ResNet34 + U-Net | **0.7257** | 0.7330 | 0.8656 | 0.8271 | 0.8459 |

The pretrained model achieved a higher validation IoU than the Week 1 scratch U-Net:

**0.7257 vs. 0.7133**

This satisfies the requirement of achieving a measurable improvement in validation IoU.

However, the scratch U-Net achieved better performance on the held-out test set.

## Week 1 Additional Experiments

### MNDWI Feature

MNDWI was calculated from the Green and SWIR1 bands and added as an additional feature.

| Metric | Result |
|---|---:|
| Validation IoU | 0.7177 |
| Test IoU | 0.7516 |
| Test Precision | 0.8621 |
| Test Recall | 0.8543 |
| Test F1 | 0.8582 |

MNDWI improved validation performance slightly compared with the baseline, but did not improve test-set performance.

### Dice + BCE

A combined Dice + BCE loss was also evaluated.

| Metric | Result |
|---|---:|
| Validation IoU | 0.7201 |
| Test IoU | 0.7756 |
| Test Precision | 0.9154 |
| Test Recall | 0.8355 |
| Test F1 | 0.8736 |

The results were very close to the baseline U-Net.

### Weighted BCE + Dice

Weighted BCE + Dice increased attention to the water class.

| Metric | Result |
|---|---:|
| Validation IoU | 0.6805 |

The experiment increased recall but also produced more false-positive predictions and therefore reduced validation IoU.

## Error Analysis

Visual error analysis showed that both the scratch and pretrained models can identify the main water structures in several images.

However, both models struggle with:

- Thin water channels
- Small water bodies
- Low-area water regions
- Difficult or low-contrast water structures

The pretrained model did not show a consistent qualitative advantage over the scratch U-Net.

## Week 2 Conclusion

In Week 2, a pretrained ResNet34 encoder with a U-Net decoder was fine-tuned for 12-channel water segmentation. The original ImageNet-pretrained first convolution was adapted from 3 input channels to 12 channels using weight averaging for the additional channels.

The pretrained U-Net achieved a validation IoU of **0.7257**, compared with **0.7133** for the Week 1 scratch U-Net. Therefore, the pretrained approach achieved a measurable improvement in validation IoU and satisfied the main performance requirement of this week.

However, on the held-out test set, the scratch U-Net achieved a higher IoU (**0.7750**) and F1-score (**0.8732**) than the pretrained U-Net (**0.7330 IoU** and **0.8459 F1**). This shows that the improvement on the validation set did not translate into better test-set generalization.

The results suggest that ImageNet pretraining provided useful initialization for the multispectral segmentation task, but adapting RGB-pretrained features to 12-channel multispectral and auxiliary data does not necessarily guarantee better generalization.

## Overall Conclusion

This project successfully developed and evaluated both a scratch U-Net and a pretrained ResNet34 + U-Net model for 12-channel water segmentation.

The 12-channel scratch U-Net provides a strong baseline for the dataset, while the pretrained ResNet34 + U-Net demonstrates how a pretrained segmentation backbone can be adapted to multispectral input.

Although the pretrained model achieved a higher validation IoU, it did not outperform the scratch model on the held-out test set. The additional Week 1 experiments with MNDWI and different loss functions also did not provide a consistent improvement over the baseline.

Overall, the experiments highlight the importance of evaluating segmentation models on a held-out test set rather than relying only on validation performance.

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385
- Chen, L.-C., et al. (2016). *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.* https://arxiv.org/abs/1606.00915
- Segmentation Models PyTorch documentation: https://smp.readthedocs.io/
- Segmentation Models PyTorch GitHub: https://github.com/qubvel-org/segmentation_models.pytorch
