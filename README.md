# Meta-RangeSeg
> Song Wang, Jianke Zhu*, Ruixiang Zhang

This is the official implementation of **Meta-RangeSeg: LiDAR Sequence Semantic Segmentation Using Multiple Feature Aggregation**  [[Paper](https://arxiv.org/pdf/2202.13377.pdf)] [[Video](https://youtu.be/xUFsmmjZYuA)]. [![arXiv](https://img.shields.io/badge/arxiv-2202.13377-b31b1b.svg)](https://arxiv.org/abs/2202.13377) 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meta-rangeseg-lidar-sequence-semantic/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=meta-rangeseg-lidar-sequence-semantic)

<p align="center"> <a><img src="fig/framework.png" width="80%"></a> </p>

|                      | **Prediction**              | **Groud Truth**        |
| -------------------- | --------------------------- | ---------------------- |
| **Perspective View** | ![z](fig/front_60_pred.gif) | ![z](fig/front_60.gif) |
| **Bird's-Eye View**   | ![z](fig/top_60_pred.gif)   | ![z](fig/top_60.gif)   |


## Demo
<p align="center"> <a href="https://youtu.be/xUFsmmjZYuA"><img src="fig/demo.png" width="80%"></a> </p>


## Model Zoo
|                                                Model                                                | Task   | mIoU(paper)<br>[on test set] | mIoU(reprod.)<br>[on test set] | Results |
|:---------------------------------------------------------------------------------------------------:| :----------------: | :----------------: | :----------------: | :----------------: |
| [Meta-RangeSeg](https://drive.google.com/file/d/1vq1fP6MjXIYZTnW6jhAAMfidKJCuZ3Zl/view?usp=sharing) | **multiple scans** semantic segmentation | 49.5 | 49.7 | [valid_pred](https://drive.google.com/file/d/1yMFevZtoZcaYZ6F6dtONKIwT9QXgSTyf/view?usp=sharing)<br>[test_pred](https://drive.google.com/file/d/1YyasP3OIALwArXqhdcvuCFG3_j0exBPm/view?usp=sharing)|
| [Meta-RangeSeg](https://drive.google.com/file/d/1k5fhZO4PYVdhkEkAnHlFje9is88yMXs-/view?usp=sharing) | **single scan** semantic segmentation | 61.0 | 60.3 | [valid_pred](https://drive.google.com/file/d/13nD-nZdlUB7sktXSnWsVDsDCn5RMAMeF/view?usp=sharing)<br>[test_pred](https://drive.google.com/file/d/1VMmoxdJqPGKIGewR_DED1K0terVHJgKG/view?usp=sharing)|


## Data Preparation
### SemanticKITTI download
Please download the original SemanticKITTI dataset from the [official website](http://www.semantic-kitti.org/dataset.html#download).

### Residual image generation
For residual image generation, we provide an online version but adopt the offline one in the actual training.
Please refer to [LiDAR-MOS](https://github.com/PRBonn/LiDAR-MOS) for more details. Thanks for their great work!


## Testing Pretrained Models
You can run the following command to test the performance of Meta-RangeSeg: 
```bash
cd ./train/tasks/semantic
python infer.py -d ./data/semantic_kitti/dataset -m ../../../logs
```


## Training
To train the model from scratch, you can run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py -d ./data/semantic_kitti/dataset -ac ../../../meta_rangeseg.yml
```


## Acknowledgment
This project is heavily based on [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) and [LiDAR-MOS](https://github.com/PRBonn/LiDAR-MOS). 
[RangeDet](https://github.com/tusen-ai/RangeDet) and [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI) are also excellent range-based models, which help us a lot.


## Citations
```
@article{wang2022meta,
  title={Meta-RangeSeg: LiDAR Sequence Semantic Segmentation Using Multiple Feature Aggregation},
  author={Wang, Song and Zhu, Jianke and Zhang, Ruixiang},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={9739--9746},
  year={2022},
  publisher={IEEE}
}
```
