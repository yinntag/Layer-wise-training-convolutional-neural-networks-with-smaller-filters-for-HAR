## Layer-wise training convolutional neural networks with smaller filters for human activity recognition using wearable sensors

This project aims to build a machine learning model for end-to-end systems to predict common human activities. Using Pytorch framework, convolutional neural nets with local loss functions to build a deep network. After the model is trained, it is saved and exported to an android application. The predictions are made using the model and get the UI interface.

This work is created by the paper. The link is https://arxiv.org/pdf/2005.03948.pdf.

Based on UCI dataset, this paper was implemented by the following steps:

1.Get UCI dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into training and test sets；

2.Construct a deep CNNs using Lego Filters；

3.Create local loss functions for minimizing the loss so that the computational graph is detached after each hidden layer to prevent standard backward gradient flow；

4.Build the whole model using the Pytorch framework to access at the end-to-end system and train the model for 500 Epochs；

5.Save the computation graph with weights, history, predictions and generate a single pt file；

6.Export the file to an android application using Android Studio and make predictions on a smartphone.
