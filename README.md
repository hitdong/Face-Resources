# Face-Resources
Following is a growing list of some of the materials I found on the web for research on face recognition algorithm.

## Papers

1. [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf).A work from Facebook.
2. [FaceNet](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf).A work from Google.
3. [ One Millisecond Face Alignment with an Ensemble of Regression Trees](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf). Dlib implements the algorithm.
4. [DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
5. [DeepID2](http://arxiv.org/abs/1406.4773)
6. [DeepID3](http://arxiv.org/abs/1502.00873)
7. [Learning Face Representation from Scratch](http://arxiv.org/abs/1411.7923)
8. [Face Search at Scale: 80 Million Gallery](http://arxiv.org/abs/1507.07242)
9. [A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)

10. [NormFace: L2 Hypersphere Embedding for Face Verification](https://arxiv.org/abs/1704.06369).* attention: model released !*
11. [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

## Datasets

1. [CASIA WebFace Database](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). 10,575 subjects and 494,414 images
2. [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).13,000 images and 5749 subjects
3. [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/) 202,599 images and 10,177 subjects. 5 landmark locations, 40 binary attributes.
4. [MSRA-CFW](http://research.microsoft.com/en-us/projects/msra-cfw/). 202,792 images and 1,583 subjects.
5. [MegaFace Dataset](http://megaface.cs.washington.edu/) 1 Million Faces for Recognition at Scale
690,572 unique people
6. [FaceScrub](http://vintage.winklerbros.net/facescrub.html). A Dataset With Over 100,000 Face Images of 530 People.
7. [FDDB](http://vis-www.cs.umass.edu/fddb/).Face Detection and Data Set Benchmark. 5k images.
8. [AFLW](https://lrs.icg.tugraz.at/research/aflw/).Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. 25k images.
9. [AFW](http://www.ics.uci.edu/~xzhu/face/). Annotated Faces in the Wild. ~1k images.
10.[3D Mask Attack Dataset](https://www.idiap.ch/dataset/3dmad). 76500 frames of 17 persons using Kinect RGBD with eye positions (Sebastien Marcel)
11. [Audio-visual database for face and speaker recognition](https://www.idiap.ch/dataset/mobio).Mobile Biometry MOBIO http://www.mobioproject.org/
12. [BANCA face and voice database](http://www.ee.surrey.ac.uk/CVSSP/banca/). Univ of Surrey
13. [Binghampton Univ 3D static and dynamic facial expression database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html). (Lijun Yin, Peter Gerhardstein and teammates)
14. [The BioID Face Database](https://www.bioid.com/About/BioID-Face-Database). BioID group
15. [Biwi 3D Audiovisual Corpus of Affective Communication](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html).  1000 high quality, dynamic 3D scans of faces, recorded while pronouncing a set of English sentences.
16. [Cohn-Kanade AU-Coded Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm).  500+ expression sequences of 100+ subjects, coded by activated Action Units (Affect Analysis Group, Univ. of Pittsburgh.
17. [CMU/MIT Frontal Faces ](http://cbcl.mit.edu/software-datasets/FaceData2.html). Training set:  2,429 faces, 4,548 non-faces; Test set: 472 faces, 23,573 non-faces.
18. [AT&T Database of Faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) 400 faces of 40 people (10 images per people)



## Trained Model

1. [openface](https://github.com/cmusatyalab/openface). Face recognition with Google's FaceNet deep neural network using Torch.
2. [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). VGG-Face CNN descriptor. Impressed embedding loss.
3. [SeetaFace Engine](https://github.com/seetaface/SeetaFaceEngine). SeetaFace Engine is an open source C++ face recognition engine, which can run on CPU with no third-party dependence. 
4. [Caffe-face](https://github.com/ydwen/caffe-face) - Caffe Face is developed for face recognition using deep neural networks. 

5. [Norm-Face](https://github.com/happynear/NormFace) - Norm Face, finetuned from  [center-face](https://github.com/ydwen/caffe-face) and [Light-CNN](https://github.com/AlfredXiangWu/face_verification_experiment)


## Tutorial

1. [Deep Learning for Face Recognition](http://valse.mmcheng.net/deep-learning-for-face-recognition/). Shiguan Shan, Xiaogang Wang, and Ming yang.

## Software

1. [OpenCV](http://opencv.org/). With some trained face detector models.
2. [dlib](http://dlib.net/ml.html). Dlib implements a state-of-the-art of face Alignment algorithm.
3. [ccv](https://github.com/liuliu/ccv).  With a state-of-the-art frontal face detector
4. [libfacedetection](https://github.com/ShiqiYu/libfacedetection). A binary library for face detection in images.
5. [SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine). An open source C++ face recognition engine.

##Frameworks

1. [Caffe](http://caffe.berkeleyvision.org/)
2. [Torch7](https://github.com/torch/torch7)
3. [Theano](http://deeplearning.net/software/theano/)
4. [cuda-convnet](https://code.google.com/p/cuda-convnet/)
5. [MXNET](https://github.com/dmlc/mxnet/)
6. [Tensorflow](https://github.com/tensorflow)
7. [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn)

## Miscellaneous

1. [faceswap](https://github.com/matthewearl/faceswap)  Face swapping with Python, dlib, and OpenCV
2. [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial) Competition on Kaggle.
3. [An implementation of Face Alignment at 3000fps via Local Binary Features](https://github.com/freesouls/face-alignment-at-3000fps)

---

>Created by betars on 27/10/2015.

人脸对齐 
Face Alignment at 3000 FPS via Regressing Local Binary Features CVPR2014 
https://github.com/yulequan/face-alignment-in-3000fps 
https://github.com/luoyetx/face-alignment-at-3000fps 
https://github.com/freesouls/face-alignment-at-3000fps

人脸对齐 
Face Alignment by Explicit Shape Regression CVPR2012 
https://github.com/soundsilence/FaceAlignment

人脸对齐 
Robust face landmark estimation under occlusion   ICCV2013 
http://www.vision.caltech.edu/xpburgos/ICCV13/

人脸对齐 
Dense Face Alignment ICCVW2017 
MatConvNet code 
model can run at real time during testing

Pose-Invariant Face Alignment with a Single CNN ICCV2017 
4.3 FPS on a Titan X GPU

http://cvlab.cse.msu.edu/project-pifa.html

二值网络人脸对齐 
Binarized Convolutional Landmark Localizers for Human Pose Estimation and Face Alignment with Limited Resources 
ICCV2017 
https://www.adrianbulat.com/binary-cnn-landmarks 
Torch7：https://github.com/1adrianb/binary-human-pose-estimation

PyTorch 2D/3D人脸对齐(特征点检测)库 
GitHub: https://github.com/1adrianb/face-alignment ​​​​

人脸对齐 性能饱和探讨 
How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) 
ICCV2017 
https://www.adrianbulat.com/face-alignment 
Pytorch Code: https://github.com/1adrianb/face-alignment 
Torch7 Code: https://github.com/1adrianb/2D-and-3D-face-alignment

人脸检测 
Face Detection with End-to-End Integration of a ConvNet and a 3D Model 
ECCV2016 
mxnet code：https://github.com/tfwu/FaceDetection-ConvNet-3D

人脸检测 速度较快 效果较好 800*1000 一个 GPU 0.1s 
SSH: Single Stage Headless Face Detector ICCV2017 
https://github.com/mahyarnajibi/SSH

人脸检测 
Face Detection with the Faster R-CNN 
https://github.com/playerkk/face-py-faster-rcnn

人脸检测 
Faceness-Net: Face Detection through Deep Facial Part Responses PAMI2017 
From Facial Parts Responses to Face Detection: A Deep Learning Approach ICCV2016 
http://shuoyang1213.me/projects/Faceness/Faceness.html 
https://pan.baidu.com/s/1qWFwqFM Password: 4q8y

人脸检测 
Face Detection through Scale-Friendly Deep Convolutional Networks 
http://shuoyang1213.me/projects/ScaleFace/ScaleFace.html

级联人脸检测 
Compact Convolutional Neural Network Cascade for Face Detection 
CEUR Workshop Proceedings, 1576, 375-387 2016 
https://github.com/Bkmz21/CompactCNNCascade

快速人脸检测对齐 效果好 
Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks 
《IEEE Signal Processing Letters》 , 2016 , 23 (10) :1499-1503 
https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html 
https://github.com/kpzhang93/MTCNN_face_detection_alignment 
https://github.com/blankWorld/MTCNN-Accelerate-Onet 
https://github.com/Seanlinx/mtcnn

级联人脸检测 
A Convolutional Neural Network Cascade for Face Detection CVPR2015 
https://github.com/anson0910/CNN_face_detection 
https://github.com/mks0601/A-Convolutional-Neural-Network-Cascade-for-Face-Detection 
https://github.com/IggyShone/fast_face_detector

人脸检测 
Recurrent Scale Approximation for Object Detection in CNN ICCV2017 
https://github.com/sciencefans/RSA-for-object-detection

人脸识别 
SphereFace: Deep Hypersphere Embedding for Face Recognition CVPR2017 
https://github.com/wy1iu/sphereface

人脸识别 
C++ 代码： https://github.com/seetaface/SeetaFaceEngine

人脸识别 
A Discriminative Feature Learning Approach for Deep Face Recognition 
code: https://github.com/ydwen/caffe-face

人脸数据库 
http://www.face-rec.org/databases/ 
https://github.com/betars/Face-Resources 
http://blog.csdn.net/chenriwei2/article/details/50631212 
http://blog.csdn.net/u012374174/article/details/71420766?locationNum=12&fps=1
