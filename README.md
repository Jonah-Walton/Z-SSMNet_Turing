# Z-SSMNet: Zonal-aware Self-Supervised Mesh Network for Prostate Cancer Detection and Diagnosis with Bi-parametric MRI

Please cite the following paper if you use the Z-SSMNet

```
@article{yuan2025z,
  title={Z-SSMNet: Zonal-aware Self-supervised Mesh Network for prostate cancer detection and diagnosis with Bi-parametric MRI},
  author={Yuan, Yuan and Ahn, Euijoon and Feng, Dagan and Khadra, Mohamed and Kim, Jinman},
  journal={Computerized Medical Imaging and Graphics},
  pages={102510},
  year={2025},
  publisher={Elsevier}
}
```

## Installation

git clone the repository
Ensure that you have a Python version after 3.10 and before 3.13 
Install the correct verions of pytorch
Example:
  #Cuda Version 12.6
  #pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## General Setup

We define setup steps that must be completed before following the algorithm tutorials.

### Data Preparation

Unless specified otherwise, this tutorial assumes that the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/) will be downloaded and unpacked. Before downloading the dataset, read its [documentation](https://zenodo.org/record/6624726) and [dedicated forum post](https://grand-challenge.org/forums/forum/pi-cai-607/topic/public-training-and-development-dataset-updates-and-fixes-631/) (for all updates/fixes, if any). To download and unpack the dataset, run the following commands:

```shell
# It might be quicker to just download the dataset to downloads and transfer
# download all folds (0-4)
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold0.zip?download=1" --output picai_public_images_fold0.zip

# unzip all folds
unzip picai_public_images_fold0.zip -d /input/images/

# Replace Download labels
git clone https://github.com/DIAGNijmegen/picai_labels {path to directory}/input/labels/
```

### Cross-Validation Splits

We use the PI-CAI challenge organizers prepared 5-fold cross-validation splits of all 1500 cases in the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/). There is no patient overlap between training/validation splits. You can load these splits as follows:

```python
from z_ssmnet.splits.picai import train_splits, valid_splits

for fold, ds_config in train_splits.items():
    print(f"Training fold {fold} has cases: {ds_config['subject_list']}")

for fold, ds_config in valid_splits.items():
    print(f"Validation fold {fold} has cases: {ds_config['subject_list']}")
```

Additionally, the organizers prepared 5-fold cross-validation splits of all cases with an [expert-derived csPCa annotation](https://github.com/DIAGNijmegen/picai_labels/tree/main/csPCa_lesion_delineations/human_expert). These splits are subsets of the splits above. You can load these splits as follows:

```python
from z_ssmnet.splits.picai_nnunet import train_splits, valid_splits
```

When using `picai_eval` from the command line, we recommend saving the splits to disk. Then, you can pass these to `picai_eval` to ensure all cases were found. You can export the labelled cross-validation splits using:

```shell
python -m z_ssmnet.splits.picai_nnunet --output "/workdir/splits/picai_nnunet"
```

### Data Preprocessing

We follow the [`nnU-Net Raw Data Archive`](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) format to prepare our dataset for usage. For this, you can use the [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep) module. Note, the [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep) module should be automatically installed when installing the `Z-SSMNet` module, and is installed within the `z-ssmnet` Docker container as well.

To convert the dataset in `/input/` into the [`nnU-Net Raw Data Archive`](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) format, and store it in `/workdir/nnUNet_raw_data`, please follow the instructions [provided here](https://github.com/DIAGNijmegen/picai_prep#mha-archive--nnu-net-raw-data-archive), or set your target paths in [`prepare_data_semi_supervised.py`](https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/prepare_data_semi_supervised.py) and execute it:

```shell
python src/z_ssmnet/prepare_data_semi_supervised.py
```

To adapt/modify the preprocessing pipeline or its default specifications, please make changes to the [`prepare_data_semi_supervised.py`](https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/prepare_data_semi_supervised.py) script accordingly.

If you want to train the supervised model (only using the data with manual labels), prepare the dataset using [`prepare_data.py`](https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/prepare_data.py) and replace `Task2302_z-nnmnet` with `Task2301_z-nnmnet` in the following commands.

## Model Training

The implementation of the model consists of three main parts:

* [Zonal Segmentation](https://github.com/yuanyuan29/Z-SSMNet/tree/master#zonal-segmentation)
* [SSL Pre-training](https://github.com/yuanyuan29/Z-SSMNet/tree/master#ssl-pre-training)
* [Z-nnMNet](https://github.com/yuanyuan29/Z-SSMNet/tree/master#z-nnmnet)

### Zonal Segmentation

The prostate area consists of the peripheral zone (PZ), transition zone (TZ), central zone (CZ) and anterior fibromuscular stroma (AFS). Prostate cancer (PCa) lesions located in different zones have different characteristics. Moreover, approximately 70%-75% of PCa originate in the PZ and 20%-30% in the TZ [[1]](https://www.sciencedirect.com/science/article/pii/S0302283815008489). In this work, we trained a standard `3D nnU-Net` [[2]](https://www.nature.com/articles/s41592-020-01008-z) with external public datasets to generate binary prostate zonal anatomy masks (peripheral and rest (TZ, CZ, AFS) of the gland) as additional input information to guide the network to learn region-specific knowledge useful for clinically significant PCa (csPCa) detection and diagnosis.

[→ Read the full documentation here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/zonal_segmentation.md).

### SSL Pre-training

SSL is a general learning framework that relies on surrogate (pretext) tasks that can be formulated using only unsupervised data. A pretext task is designed in a way that solving it requires learning of valuable image representations for the downstream (main) task, which contributes to improving the generalization ability and performance of the model. We introduced image restoration as the pretext task and pre-trained our zonal-aware mesh network in a self-supervised manner.

[→ Read the full documentation here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/ssl.md).

### Z-nnMNet

Considering the heterogeneous between data from multi-centres and multi-vendors, we integrated the zonal-aware mesh network into the famous nnU-Net framework, which provides a performant framework for medical image segmentation to form the `Z-nnMNet` that can pre-process the data adaptively. For large datasets with labels, the model can be trained from scratch. If the dataset is small or some labels of the data are noisy, fine-tuning based on the SSL pre-trained model can help achieve better performance.

[→ Read the full documentation here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/z-nnmnet.md).

## References

[[1]](https://www.sciencedirect.com/science/article/pii/S0302283815008489) J. C. Weinreb, J. O. Barentsz, P. L. Choyke, F. Cornud, M. A. Haider, K. J. Macura, D. Margolis, M. D. Schnall, F. Shtern, C. M. Tempany, H. C. Thoeny, and S. Verma, “PI-RADS Prostate Imaging - Reporting and Data System: 2015, Version 2,” European Urology, vol. 69, no. 1, pp. 16-40, Jan, 2016.

[[2]](https://www.nature.com/articles/s41592-020-01008-z) F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203-+, Feb, 2021.

[[3]](https://univ-rennes.hal.science/hal-04393722/) Z. Dong, Y. He, X. Qi, Y. Chen, H. Shu, J.-L. Coatrieux, G. Yang, and S. Li, “MNet: Rethinking 2D/3D Networks for Anisotropic Medical Image Segmentation,” Thirty-First International Joint Conference on Artificial Intelligence {IJCAI-22}, Jul 2022, Vienna, Austria. pp.870-876.

[[4]](https://www.sciencedirect.com/science/article/pii/S1361841520302048) Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, “Models Genesis,” Med Image Anal, vol. 67, pp. 101840, Jan, 2021.

[[5]](https://zenodo.org/record/6667655) A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655


the command nnunet runs nnunet_wrapper
the command train runs train_z_ssmnet