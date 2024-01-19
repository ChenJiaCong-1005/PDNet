# Lightweight Parameter De-redundancy Demoireing Network with Adaptive Wavelet Distillation

### [Dataset](https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e?usp=sharing) | [Paper](https://link.springer.com/article/10.1007/s11554-023-01386-5)

**Lightweight Parameter De-redundancy Demoireing Network with Adaptive Wavelet Distillation** (JRTIP 2024)  
Jiacong Chen, Qingyu Mao, Youneng Bao, Yingcong Huang, Fanyang Meng, Yongsheng Liang



## Abstract
Recently, deep convolutional neural networks (CNNs) have achieved significant advancements in single image demoiréing. However, most of the existing CNN-based demoiréing methods require excessive memory usage and computational cost, which considerably limit to apply on mobile devices. Additionally, most CNN-based methods employ expensive approaches to generate similar feature maps, thereby resulting in redundant parameters within these networks. To alleviate these issues, we propose a lightweight parameter de-redundancy network (PDNet) for image demoiréing. Specifically, we present an efficient ghost block (EGB) that utilizes dilated convolution and cost-efficient operation, which significantly reduces parameters and extracts representative features. Meanwhile, we design a multi-scale shuffle fusion mechanism (MSFM) with a low amount of parameters to integrate different scales of features, which mitigates the information loss issue. To enable the lightweight network for learning latent moiré removal knowledge better, we adopt adaptive wavelet distillation to guide the network training. Experimental results validate the efficacy of our proposed method, achieving comparable or even superior results to the state-of-the-art method, while utilizing only 26.32% of parameters and 29.59% of Macs. 



## Environments

First you have to make sure that you have installed all dependencies. To do so, you can create an anaconda environment  using

```
conda env create -f environment.yaml
conda activate esdnet
```

Our implementation has been tested on one NVIDIA 3090 GPU with cuda 11.2.



## Dataset
![Data](./figures/dataset.png)
We provide the 4K dataset UHDM for you to evaluate a pretrained model or train a new model.
To this end, you can download them [here](https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e?usp=sharing), 
or you can simply run the following command for automatic data downloading:
```
bash scripts/download_data.sh
```
Then the dataset will be available in the folder `uhdm_data/`.

## Download Teacher Model:
We provide pre-trained teacher model on UHDM dataset, which can be downloaded through the following link:

Link：https://pan.baidu.com/s/1mzPbs5VaFyBbPsS3aBCj5g 
Password：fqei

## Train
To train a model from scratch, simply run:

```
python train.py --config CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.
We have included configuration files for each dataset under the folder `config/`.

For example, if you want to train our lightweight model ESDNet on UHDM dataset, run:
```
python train.py --config ./config/uhdm_config_distillation.yaml
```
   

## Test
To test a model, you can also simply run:

```
python test.py --config CONFIG.yaml
```

where you need to specify the value of `TEST_EPOCH` in the `CONFIG.yaml` to evaluate a model trained after specific epochs, 
or you can also specify the value of `LOAD_PATH` to directly load a pre-trained checkpoint.



## Citation
Please consider :grimacing: staring this repository and citing the following papers if you feel this repository useful.

```
@article{chen2024lightweight,
  title={Lightweight parameter de-redundancy demoir{\'e}ing network with adaptive wavelet distillation},
  author={Chen, Jiacong and Mao, Qingyu and Bao, Youneng and Huang, Yingcong and Meng, Fanyang and Liang, Yongsheng},
  journal={Journal of Real-Time Image Processing},
  volume={21},
  number={1},
  pages={6},
  year={2024},
  publisher={Springer}
}

@inproceedings{Yu2022TowardsEA,
  title={Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing},
  author={Xin Yu and Peng Dai and Wenbo Li and Lan Ma and Jiajun Shen and Jia Li and Xiaojuan Qi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

```

## Acknowledgement
Thanks to Xin Yu for his excellent work and implementation ESDNet

## Contact
If you have any questions, you can email me (2210434045@email.szu.edu.cn).


