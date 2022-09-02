[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)]()
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub license](https://img.shields.io/github/license/HuaizhengZhang/Awesome-System-for-Machine-Learning.svg?color=blue)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/LICENSE)

# Awesome-3D-Semantic-Segmentation
A curated list of research in 3D Semantic Segmentation(**Lidar-based Method**). 

You are very welcome to pull request to update this list. :smiley:   
![3D Semantic Segmentation](https://github.com/TianhaoFu/Awesome-3D-Semantic-Segmentation/blob/main/3d_seg.png)

## Dataset
- [SemanticKitti Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
  - 3,712 training samples
  - 3,769 validation samples
  - 7,518 testing samples
- [Waymo Dataset](https://waymo.com/open/)
## Top conference & workshop
### Conferene
- Conference on Computer Vision and Pattern Recognition(CVPR)
- International Conference on Computer Vision(ICCV)
- European Conference on Computer Vision(ECCV)
### Workshop
- CVPR 2019 Workshop on Autonomous Driving([nuScenes 3D detection](http://cvpr2019.wad.vision/))
- CVPR 2020 Workshop on Autonomous Driving([BDD1k 3D tracking](http://cvpr2020.wad.vision/))
- CVPR 2021 Workshop on Autonomous Driving([waymo 3D detection](http://cvpr2021.wad.vision/))
- CVPR 2022 Workshop on Autonomous Driving([waymo 3D detection](http://cvpr2022.wad.vision/))
- [CVPR 2021 Workshop on 3D Vision and Robotics](https://sites.google.com/view/cvpr2021-3d-vision-robotics)
- [CVPR 2021 Workshop on 3D Scene Understanding for Vision, Graphics, and Robotics](https://scene-understanding.com/)

- [ICCV 2019 Workshop on Autonomous Driving](http://wad.ai/)
- [ICCV 2021 Workshop on Autonomous Vehicle Vision (AVVision)](https://avvision.xyz/iccv21/), [note](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Fan_Autonomous_Vehicle_Vision_2021_ICCV_Workshop_Summary_ICCVW_2021_paper.pdf)
- [ICCV 2021 Workshop SSLAD Track 2 - 3D Object Detection](https://competitions.codalab.org/competitions/33236#learn_the_details)
- [ECCV 2020 Workshop on Commands for Autonomous Vehicles](https://c4av-2020.github.io/)
- [ECCV 2020 Workshop on Perception for Autonomous Driving](https://sites.google.com/view/pad2020)
## Paper (Lidar-based method)

### Traditinal methods
- PyramNet: Point cloud pyramid attention network and graph embedding module for classification and segmentation
- Fast semantic segmentation of 3D point clouds using a dense CRF with learned parameters
- Shape-based recognition of 3d point clouds in urban environments
- Fast semantic segmentation of 3d point clouds with strongly varying density
- Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers
- Discriminative learning of markov random fields for segmentation of 3D scan data
- Robust 3D scan point classification using associative markov networks
- Contextual classification with functional max-margin markov networks
### Point-based methods
##### Point-wise shared MLP
- PointNet: Deep learning on point sets for 3D classification and segmentation
- PointNet++: Deep hierarchical feature learning on point sets in a metric space
- PointSIFT: A SIFT-like network module for 3D point cloud semantic segmentation
- Know what your neighbors do: 3D semantic segmentation of point clouds
- RandLA-Net: Efficient semantic segmentation of large-scale point clouds
- Modeling point clouds with self-attention and gumbel subset sampling
- LSANet: Feature learning on point sets by local spatial attention
- PyramNet: Point cloud pyramid attention network and graph embedding module for classification and segmentation
##### Point Convolution
- PointCNN: Convolution on x-transformed points
- A-CNN: Annularly convolutional neural networks on point clouds
- KPConv: Flexible and deformable convolution for point clouds
- Dilated point convolutions: On the receptive field of point convolutions
- PointAtrousNet: Point atrous convolution for point cloud analysis
- PointAtrousGraph: Deep hierarchical encoder-decoder with atrous convolution for point clouds
- Tangent convolutions for dense prediction in 3D
- DAR-Net: Dynamic aggregation network for semantic scene segmentation
- ShellNet: Efficient point cloud convolutional neural networks using concentric shells statistics
- Point-voxel cnn for efficient 3D deep learning
##### Recurrent Neural Networ
- Exploring spatial context for 3D semantic segmentation of point clouds
- Recurrent slice networks for 3D segmentation of point clouds
##### Lattice Convolution
- SplatNet: Sparse lattice networks for point cloud processing
- LatticeNet: Fast point cloud segmentation using permutohedral lattices

### Voxel-based methods
- Sparse single sweep lidar point cloud segmentation via learning contextual shape priors from scene completion [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16419), [code](https://github.com/yanx27/JS3C-Net)
- Point cloud labeling using 3D convolutional neural network
- Segcloud: Semantic segmentation of 3D point cloud
- Fully-convolutional point networks for large-scale point clouds
- 3DCNN-DQN-RNN: A deep reinforcement learning framework for semantic parsing of large-scale 3D point clouds
- 3D semantic segmentation with submanifold sparse convolutional networks
- Efficient convolutions for real-time semantic segmentation of 3D point clouds
- VV-Net: Voxel vaenet with group convolutions for point cloud segmentation
- VolMap: A real-time model for semantic segmentation of a LiDAR surrounding view
### Image-based methods
##### Range view-based methods
- SqueezeSeg: Convolutional neural nets with recurrent CRF for real-time road-object segmentation from 3D LiDAR point cloud
- SqueezeSegV2: Improved model structure and unsupervised domain adaptation for road-object segmentation from a LiDAR point cloud
- SqueezeSegV3: Spatially-adaptive convolution for efficient point-cloud segmentation
- Semantic segmentation of 3D LiDAR data in dynamic scene using semi-supervised learning
- RangeNet++: Fast and accurate LiDAR semantic segmentation
- LU-Net: An efficient network for 3D LiDAR point cloud semantic segmentation based on end-to-end-learned 3D features and U-Net
- 3D-MiniNet: Learning a 2D representation from point clouds for fast and efficient 3D LiDAR semantic segmentation
- DeepTemporalSeg: Temporally consistent semantic segmentation of 3D LiDAR scans
- LiSeg: Lightweight road-object semantic segmentation in 3D LiDAR scans for autonomous driving
- PointSeg: Real-time semantic segmentation based on 3D LiDAR point cloud
- RIU-Net: Embarrassingly simple semantic segmentation of 3D LiDAR point cloud
- SalsaNet: Fast road and vehicle segmentation in LiDAR point clouds for autonomous driving
- SalsaNext: Fast,uncertainty-aware semantic segmentation of LiDAR point clouds
##### Multi view-based methods
- Deep projective 3D semantic segmentation
- Unstructured point cloud semantic labeling using deep segmentation networks
### Graph-based Methods
- Large-scale point cloud semantic segmentation with superpoint graphs
- Graph attention convolution for point cloud semantic segmentation
- Hierarchical point-edge interaction network for point cloud semantic segmentation
- Dynamic graph CNN for learning on point clouds

## Survey
- Linking Points With Labels in 3D: A Review of Point Cloud Semantic Segmentation [paper](https://arxiv.org/abs/1908.08854)
- Are We Hungry for 3D LiDAR Data for Semantic Segmentation? A Survey and Experimental Study [paper](https://arxiv.org/abs/2006.04307)
- A Technical Survey and Evaluation of Traditional Point Cloud Clustering Methods for LiDAR Panoptic Segmentation [paper](https://arxiv.org/abs/2108.09522)
- A survey on deep learning-based precise boundary recovery of semantic segmentation for images and point clouds [paper] (https://www.sciencedirect.com/science/article/pii/S0303243421001185)
