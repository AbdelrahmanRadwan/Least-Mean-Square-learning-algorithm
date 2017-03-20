# Neural-Networks-Implementations2
-  Objective:
1.  Implement the Least Mean Square learning algorithm on a single layer neural 
networks, which can be able to classify a stream of input data to one of a set of predefined 
classes.
  Use the iris data in both your training and testing processes. (Each class has 50 samples: 
train NN with the first 30 samples, and test it with the remaining 20 samples)
2.  After training, 
  Draw a line that can discriminate between the two learned classes.
  Test the classifier with the remaining 20 samples of each selected classes and find 
confusion matrix and compute overall accuracy.
-  Single layer neural network:
1. Input:
  Select two features 
  Select two classes (C1 & C2 or C1 & C3 or C2 & C3)
  Enter learning rate (eta)
  Enter number of epochs (m)
  Enter MSE threshold (mse_threshold)
  Add bias or not (Checkbox)
2. Initialization:
  Number of features = 2. 
  Number of classes = 2. 
  Weights + Bias = small random numbers or Zeros
3. Classification: 
  Sample (single sample to be classified).
4. Workflow:
  Training Phase: (repeat the following m epochs)
Assuming that we have n training samples  {ݏܽ݉݌݈݁
௜
:	݅ = 1 → ࢔}
  Fetch features (x) of  ݏܽ݉݌݈݁
௜
, and its desired output (d)
  Calculate the net value (v),
  Calculate actual output (y) using Linear activation function,
  Calculate the error = d – y,
Xi
Xj
1
y (Class ID) 
2
  Update the weights (new weights = old weights + eta * error * x), note: old weights is  ൥
ܾ
ܹ1݅
ܹ1݆
൩
  Draw line:  line equation is W1i * Xi + W1j * Xj + b = 0
  Testing Phase:
1.  Given a sample x
2.  Calculate the net value (v),
3.  Calculate actual output (y) using signum activation function,
4.  Output: y (Class ID).
  Evaluation: build the confusion matrix and overall accuracy.  
