# Metastatic Cancer Detection
In this github we open-source a pytorch pipeline we have developed for transfer learning and ensemble learning for metastatic cancer detection.

## Metastatic Cancer
In metastatic cancer (also known as stage IV of cancer), cancer cells can spread to distant parts of the body and in some situations, experts might not be able to identify where the cancer initially started. Once cancer expands, it can be hard to control, however there are still treatments that can stop, slow the growth of cancer or relieve symptoms caused by cancer.

## PCam Dataset
We are using [PCam dataset](https://github.com/basveeling/pcam) to train/validate/test our models for metastatic cancer detection. [PCam dataset](https://github.com/basveeling/pcam) consists of 327,680 color images (96 x 96px) which contains repetition. A cleaned version of PCam dataset is available in a [Kaggel competition](https://www.kaggle.com/c/histopathologic-cancer-detection/data). PCam dataset casts metastatic cancer detection as an image classification task.

![alt text](https://github.com/basveeling/pcam/blob/master/pcam.jpg "PCam image samples")

We randomly select 1000 images from PCam dataset for testing and report all measurements on selected data. We divide remainder of data into 90% training and 10% validation. We have not observed any noticable difference using 75% training and 25% validation, however we are open to share our pretrained models in both scenarios.

## How to run code?
This section and all scripts will be shared by the time of GHC 2019 conference.

### Reproducibility
The existing scripts show how multiple models can be used to make predictions in an ensemble fashion. Please contact us if you are interested to access our pretrained models. 
