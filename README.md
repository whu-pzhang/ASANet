<h2 align="center">ASANet: Asymmetric Semantic Aligning Network for RGB and SAR image land cover classification</h2>

<h5 align="right">by Zhang Pan, Baochai Peng, Chaoran Lu and Quanjin Huang</h5>


<div align="center">
  <img src="https://raw.githubusercontent.com/whu-pzhang/ASANet/main/assets/ASANet_arch.jpg"><br><br>
</div>


This is an official implementation of ASANet in our ISPRS paper [ASANet: Asymmetric Semantic Aligning Network for RGB and SAR image land cover classification](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003630).

[arXiv]()

>Synthetic Aperture Radar (SAR) images have proven to be a valuable cue for multimodal Land Cover Classification (LCC) when combined with RGB images. Most existing studies on cross-modal fusion assume that consistent feature information is necessary between the two modalities, and as a result, they construct networks without adequately addressing the unique characteristics of each modality. In this paper, we propose a novel architecture, named the Asymmetric Semantic Aligning Network (ASANet), which introduces asymmetry at the feature level to address the issue that multi-modal architectures frequently fail to fully utilize complementary features. The core of this network is the Semantic Focusing Module (SFM), which explicitly calculates differential weights for each modality to account for the modality-specific features. Furthermore, ASANet incorporates a Cascade Fusion Module (CFM), which delves deeper into channel and spatial representations to efficiently select features from the two modalities for fusion. Through the collaborative effort of these two modules, the proposed ASANet effectively learns feature correlations between the two modalities and eliminates noise caused by feature differences. Comprehensive experiments demonstrate that ASANet achieves excellent performance on three multimodal datasets. Additionally, we have established a new RGB-SAR multimodal dataset, on which our ASANet outperforms other mainstream methods with improvements ranging from 1.21% to 17.69%. The ASANet runs at 48.7 frames per second (FPS) when the input image is 256 × 256 pixels.



## Get Started

### install

1. Requirements

* Python 3.8+
* PyTorch 1.10.0 or higher
* CUDA 11.1 or higher


2. Install all dependencies. Install pytorch, cuda and cudnn, then install other dependencies via:

```
pip install -r requirements.txt
```

### Prepare Datasets

1. PIE-RGB-SAR dataset download links [Quark](https://pan.quark.cn/s/383b348cbbea) or [Google Drive](https://drive.google.com/file/d/1O7gNoRTHfxM7ih3CJprvlBijqwYccn2C/view?usp=sharing)
2. [WHU-RGB-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
3. [DDHRNet](https://github.com/XD-MG/DDHRNet/tree/main)


The structure of the data file should be like:

```shell
<datasets>
|-- <DatasetName1>
    |-- <RGBFolder>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <SARFolder>
        |-- <name1>.<ModalXFormat>
        |-- <name2>.<ModalXFormat>
        ...
    |-- <LabelFolder>
        |-- <name1>.<LabelFormat>
        |-- <name2>.<LabelFormat>
        ...
    |-- train.txt
    |-- val.txt
|-- <DatasetName2>
|-- ...
```


`train.txt` contains the names of items in training set, e.g.:

```shell
<name1>
<name2>
...
```
### Training

1. Config

    Edit config file in `configs`, including dataset and network settings.

2. Run multi GPU distributed training:
 
```shell
CUDA_VISIBLE_DEVICES="GPU IDs" bash dist_train.sh ${config} ${GPU_NUM} [optional arguments]
```

### Evaluation

Testing on a single GPU

```shell
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

## Result

| Model          | Year | FLOPs | Parameter | Speed  | mIoU          |           |               |
| -------------- | ---- | ----- | --------- | ------ | ------------- | --------- | ------------- |
|                |      | G     | *M*       | *FPS*  | *PIE-RGB-SAR* | *DDHR-SK* | *WHU-OPT-SAR* |
| FuseNet        | 2017 | 66    | *55*      | *88.8* | 60.62         | 48.87     | 38.01         |
| SA-Gate        | 2020 | 46    | 121       | 34.9   | 73.84         | 90.89     | 53.17         |
| AFNet          | 2021 | 65    | 356       | 35.9   | 76.27         | 91.11     | 53.57         |
| CMFNet         | 2022 | 77    | 104       | 21.6   | 76.31         | 89.79     | 53.72         |
| CMX            | 2023 | *15*  | *67*      | 33.5   | *77.10*       | *94.32*   | *55.68*       |
| FTransUNet     | 2024 | 70    | 203       | 20.7   | 75.72         | 87.64     | 54.47         |
| *ASANet(ours)* |      | *25*  | 82        | *48.7* | *78.31*       | *94.48*   | *56.11*       |

