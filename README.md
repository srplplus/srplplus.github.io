# Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Samples

## Overview
This project introduces an innovative approach to open-set speaker identification, leveraging rapid tuning techniques with speaker reciprocal points and the strategic use of negative samples. By refining the identification process, our method achieves significant improvements in accuracy and efficiency, setting a new benchmark in speaker recognition technology.

<p align="center">
  <img src="images/srpl_arch.png" alt="SRPL+ Architecture" width="50%" />
</p>
<p align="center">
  <img src="images/srpl.png" alt="SRPL+ Process" width="50%" />
</p>
## Dataset
We utilize two primary datasets in our research:

- **Qualcomm Speech**:

[Link to Qualcomm Speech dataset](https://developer.qualcomm.com/project/keyword-speech-dataset)

- **FFSVC HiMia**:

[Link to HiMia dataset](https://aishelltech.com/wakeup_data)

## Pretrained Audio Large Model
Our methodology is built upon a pretrained audio large model WavLM-base-plus for TDNN speaker verification, specifically designed to capture the nuances of human speech and speaker characteristics. This model serves as the foundation for our rapid tuning process, allowing for effective speaker identification.

[Link and Details to the pretrained WavLM-TDNN AudioLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)

<p align="center">
  <img src="images/wavlm.png" alt="SRPL+ Architecture" width="50%" />
</p>

## Evaluations
The evaluation section details the performance metrics, comparison with baseline models, and the impact of our proposed enhancements on open-set speaker identification. We provide a comprehensive analysis to demonstrate the superiority of our approach.

[Evaluation metrics implementation](https://github.com/srplplus/srplplus.github.io)

## Code
Code used in this research for model training, and evaluation, is available for public use after publication. This encourages reproducibility and further experimentation in the field.

**Please note**: The model architecture and evaluation code for results reproduction are released. The training code for SRPL+ will be **released after the publication** of our research paper.

[SRPL+ code repository](https://github.com/srplplus/srplplus.github.io)

## Visualization and Evaluations
We present a series of visualizations and detailed evaluations to illustrate our method's effectiveness. The embedding plots clearly demostrate the effectiveness of our method.

![emb plot](images/emb_srpl.png)

<!-- [Link to visualizations and detailed evaluations]() -->

<!-- ## How to Use
This section provides a step-by-step guide on how to replicate our research findings, including setting up the environment, preprocessing the data, training the model, and conducting evaluations. -->

## Citation
Please cite our work if it contributes to your research:

@article{srplplus2024,
title={Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Samples},
author={Anonymous Authors},
journal={Anonymous Journal},
year={2024},
publisher={Anonymous Publisher}
}
