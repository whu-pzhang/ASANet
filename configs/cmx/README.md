# CMX

> [CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers](http://arxiv.org/abs/2203.04838)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/huaaaliu/RGBX_Semantic_Segmentation/tree/main">Official Repo</a>


## Abstract

<!-- [ABSTRACT] -->

Scene understanding based on image segmentation is a crucial component of autonomous vehicles. Pixel-wise semantic segmentation of RGB images can be advanced by exploiting complementary features from the supplementary modality (Xmodality). However, covering a wide variety of sensors with a modality-agnostic model remains an unresolved problem due to variations in sensor characteristics among different modalities. Unlike previous modality-specific methods, in this work, we propose a unified fusion framework, CMX, for RGB-X semantic segmentation. To generalize well across different modalities, that often include supplements as well as uncertainties, a unified crossmodal interaction is crucial for modality fusion. Specifically, we design a Cross-Modal Feature Rectification Module (CM-FRM) to calibrate bi-modal features by leveraging the features from one modality to rectify the features of the other modality. With rectified feature pairs, we deploy a Feature Fusion Module (FFM) to perform sufficient exchange of long-range contexts before mixing. To verify CMX, for the first time, we unify five modalities complementary to RGB, i.e., depth, thermal, polarization, event, and LiDAR. Extensive experiments show that CMX generalizes well to diverse multi-modal fusion, achieving state-of-the-art performances on five RGB-Depth benchmarks, as well as RGBThermal, RGB-Polarization, and RGB-LiDAR datasets. Besides, to investigate the generalizability to dense-sparse data fusion, we establish an RGB-Event semantic segmentation benchmark based on the EventScape dataset, on which CMX sets the new state-of-the-art. The source code of CMX is publicly available at https://github.com/huaaaliu/RGBX_Semantic_Segmentation.


