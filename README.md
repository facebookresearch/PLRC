## Point-Level Region Contrast for Object Detection Pre-Training


This is a PyTorch implementation of the [PLRC paper](https://arxiv.org/abs/2202.04639):
```
@inproceedings{bai2022point,
  title={Point-Level Region Contrast for Object Detection Pre-Training},
  author={Bai, Yutong and Chen, Xinlei and Kirillov, Alexander and Yuille, Alan and Berg, Alexander C},
  booktitle={CVPR},
  year={2022}
}
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).



### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_plrc.py \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 
```
This script uses all the default hyper-parameters as described in the PRLC paper.


### Models

Our pre-trained ResNet-50 model and finetuned checkpoints on object detection can be downloaded as following:


|             |                                          Pretrained Model                                           | Epoch | 
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: 
| Res50   | [download link](https://dl.fbaipublicfiles.com/plrc/pre-train/model_final.pth) |   100   | 



|             |                                          Finetuned Model                                           |  AP | AP50 | AP75 |
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: | :--------: | :--------: |
| Res50   | [download link](https://dl.fbaipublicfiles.com/plrc/fine-tune/model_final.pth) |   58.2   |    82.7     |    65.1    |

The APs on Pascal VOC is averaged over 5 times.


### Detection

Same as [MoCo](https://github.com/facebookresearch/moco) for object detection transfer, please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).


### Visualization

For model visualzation, we provide an [google colab](https://colab.research.google.com/drive/172dmSGYAzEgiMJ1RFyuStrj_YOQVvFpQ?usp=sharing) for better illustration.




### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
