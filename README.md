# Synthetic Data Generation in Medical Applications
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/helenmand/project-in-advanced-topics-in-ML-DWS-SS24/blob/main/assets/wgangpgif2.gif" alt="First GIF" style="width: 48%;">
  <img src="https://github.com/helenmand/project-in-advanced-topics-in-ML-DWS-SS24/blob/main/assets/wgangpgif2.gif" alt="Second GIF" style="width: 48%;">
</div>

![](https://github.com/helenmand/project-in-advanced-topics-in-ML-DWS-SS24/blob/main/assets/wgangpgif2.gif)

## About
In our project, "Synthetic Data Generation in Medical Applications," we explored the generation of synthetic datasets to address the challenges of privacy and accessibility in healthcare data. 
The dataset we used are the following:

| Type       | Dataset Name                                            
|------------|---------------------------------------------------------
| **Tabular**| [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)|
|            | [Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)|
| **Timeseries** | [Biosignals for estimating mental concentration](https://ieee-dataport.org/open-access/baseline-dataset-biosignals-estimating-mental-concentration)      |
|            | [Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)          |
| **Images** |[KneeXrayOA-simple](https://www.kaggle.com/datasets/tommyngx/kneexrayoa-simple?resource=download) |
|            |[ChestXRay Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)        |

Our methodology included:
- for Tabular Data
  - Advanced Methods like GANs (CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer, and medGAN)
  - Statistical Methods (SMOTE, GaussianCopulaSynthesizer) 

- for Time series:
  - PARSynthesizer 

- for Images:
  - WGAN and
  - WGAN-GP
---


