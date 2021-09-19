# maskDetector
口罩识别系统

本项目为口罩识别系统，给定照片，圈出对应人脸，绿色框表示佩戴口罩，红色框表示并未佩戴口罩
 
项目简介：

深度学习框架：Pytorch

模型：使用MTCNN多任务卷积神经网络进行人脸判断+VGGNet经典神经网络模型进行对应人脸口罩佩戴情况二分类 

Attention!! /weights下给出训练好的MTCNN模型（三个网络）。由于VGGNet参数过多，无法直接在仓库中存储，故在一个新release中上传vgg.pth，将其放置于/weights文件夹下即可运行
