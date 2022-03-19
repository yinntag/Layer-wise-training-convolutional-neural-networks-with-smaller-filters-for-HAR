# Layer-wise training convolutional neural networks with smaller filters for human activity recognition using wearable sensors
![Image text](https://github.com/yinntag/Layer-wise-training-convolutional-neural-networks-with-smaller-filters-for-HAR/blob/master/Figure/model.png)
# Abstract
Recently, convolutional neural networks (CNNs) have set latest state-of-the-art on various human activity recognition (HAR) datasets. However, deep CNNs often require more computing resources, which limits their applications in embedded HAR. Although many successful methods have been proposed to reduce memory and FLOPs of CNNs, they often involve special network architectures designed for visual tasks, which are not suitable for deep HAR tasks with time series sensor signals, due to remarkable discrepancy. Therefore, it is necessary to develop lightweight deep models to perform HAR. As filter is the basic unit in constructing CNNs, it deserves further research whether re-designing smaller filters is applicable for deep HAR. In the article, inspired by the idea, we proposed a lightweight CNN using Lego filters for HAR. A set of lower-dimensional filters is used as Lego bricks to be stacked for conventional filters, which does not rely on any special network structure. The local loss function is used to train model. To our knowledge, this is the first paper that proposes lightweight CNN for HAR in ubiquitous and wearable computing arena. The experiment results on five public HAR datasets, UCI-HAR dataset, OPPORTUNITY dataset, UNIMIB-SHAR dataset, PAMAP2 dataset, and WISDM dataset collected from either smartphones or multiple sensor nodes, indicate that our novel Lego CNN with local loss can greatly reduce memory and computation cost over CNN, while achieving higher accuracy. That is to say, the proposed model is smaller, faster and more accurate. Finally, we evaluate the actual performance on an Android smartphone.
# Attention
This project aims to build a machine learning model for end-to-end systems to predict common human activities. Using Pytorch framework, Lego filters with local loss functions to build a convolutional network. After the model is trained, it is saved and exported to an android application. The predictions are made using the model and get the UI interface. This work is created by the paper. The link is https://arxiv.org/pdf/2005.03948.pdf.

# Requirements
- python 3
- pytorch >= 1.1.0
- torchvision
- numpy 1.21

# Usage
Based on UCI dataset, this paper was implemented by the following steps:

1. Get UCI dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into training and test sets；
- Run dataset preprocessing.py to get them.

2. Construct a deep CNNs using Lego Filters；
- Run train.py to using Lego networks alone.

3. Create local loss functions for minimizing the loss so that the computational graph is detached after each hidden layer to prevent standard backward gradient flow；

4. Build the whole model using the Pytorch framework to access at the end-to-end system and train the model for 500 Epochs；
- Run train_local loss.py to using Lego networks with local loss functions.

5. Save the computation graph with weights, history, predictions and generate a single pt file；

6. Export the file to an android application using Android Studio and make predictions on a smartphone.

# Contributing
We appreciate all contributions. Please do not hesitate to let me know if you have any problems during the reproduction.

# Citation
If you find it useful in your research, please consider citing.

@article{tang2020layer,
  title={Layer-wise training convolutional neural networks with smaller filters for human activity recognition using wearable sensors},
  author={Tang, Yin and Teng, Qi and Zhang, Lei and Min, Fuhong and He, Jun},
  journal={IEEE Sensors Journal},
  volume={21},
  number={1},
  pages={581--592},
  year={2020},
  publisher={IEEE}
}
