# Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Samples

## Overview
This project introduces an innovative approach to open-set speaker identification, leveraging rapid tuning techniques with speaker reciprocal points and the strategic use of negative samples. By refining the identification process, our method achieves significant improvements in accuracy and efficiency, setting a new benchmark in speaker recognition technology.

## Dataset
We utilize two primary datasets in our research:
- **Qualcomm Speech**: (Here, provide a brief description of the Snapdragon dataset, including its characteristics and why it's suitable for this research.)

[Link to Qualcomm dataset]()

- **FFSVC HiMia**: (Provide details about the HeyMiya dataset, highlighting its unique features and contribution to the research.)

[Link to HiMia dataset]()

## Pretrained Audio Large Model
Our methodology is built upon a pretrained audio large model WavLM-base-plus for TDNN speaker verification, specifically designed to capture the nuances of human speech and speaker characteristics. This model serves as the foundation for our rapid tuning process, allowing for effective speaker identification.

[Link to pretrained AudioLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)

![Self-supervised learning (SSL) WavLM](images/wavlm.png "Self-supervised learning (SSL) WavLM")


## Evaluations
The evaluation section details the performance metrics, comparison with baseline models, and the impact of our proposed enhancements on open-set speaker identification. We provide a comprehensive analysis to demonstrate the superiority of our approach.

## Code
Code used in this research for model training, and evaluation, is available for public use after publication. This encourages reproducibility and further experimentation in the field.

### SRPL+ Training Code Release

**Please note**: The training code for SRPL+ will be **released after the publication** of our research paper. This decision ensures that we provide a comprehensive and fully reviewed codebase that aligns with the final published methods and results.

[Link to code repository]()

## Visualization and Evaluations
We present a series of visualizations and detailed evaluations to illustrate our method's effectiveness. These include confusion matrices, ROC curves, and comparison charts that highlight the improvements over existing techniques.

[Link to visualizations and detailed evaluations]()

## How to Use
This section provides a step-by-step guide on how to replicate our research findings, including setting up the environment, preprocessing the data, training the model, and conducting evaluations.

## Citation
Please cite our work if it contributes to your research:

@article{srplplus2024,
title={Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Samples},
author={Anonymous Authors},
journal={Anonymous Journal},
year={2024},
publisher={Anonymous Publisher}
}
