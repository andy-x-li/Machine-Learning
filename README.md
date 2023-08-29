# Machine-Learning
A collection of Machine Learning Projects from my time at University.

Kernel Regression is a technique used in machine learning to address the nonlinearity present in some datasets. In `python Kernel_Regression.py`, I used a Gaussian kernel to train a kernel ridge regression classifier (K). K is calculated using the following equation:

<p align="center">
  <img width="260" alt="Screen Shot 2023-08-28 at 10 03 14 PM" src="https://github.com/andy-x-li/Machine-Learning/assets/125074849/14698757-fb82-417b-8b94-4eec28b3eefb">
</p>

We can use the following equation to estimate α: 

<p align="center">
  <img width="186" alt="Screen Shot 2023-08-28 at 9 52 17 PM" src="https://github.com/andy-x-li/Machine-Learning/assets/125074849/d62dc297-86ad-49b0-8186-dd789480ba42">
</p>

For a new sample x, the predicted label can be estimated using the following equation:

<p align="center">
  <img width="160" alt="Screen Shot 2023-08-28 at 11 24 01 PM" src="https://github.com/andy-x-li/Machine-Learning/assets/125074849/084968fd-42b6-4465-acbb-6a9d3c924c01">
</p>

We first solve K and α from the training data (X) and then estimate the predicted label (y2). We can then estimate the error between the predicted label (y2) and the training data (y). We can also calculate the value of K for the test data and use it to estimate the predicted label (ytest2) for the test data. The error is calculated between the predicted label (ytest2) and the test data (ytest).  
 
In the code below, we use different σ^2 values (0.025, 0.05, 0.1, 0.5), and plot the error for the training data and test data for each σ^2.

To run this code: <br>
> **1)** Run via cmd line `python Kernel_Regression.py` <br>
> **2)** One-by-one, observe the graphs that pop up

**Note**: *The first window you will see displays two graphs side by side that are almost identical. The graph on the left represents the training data we randomly generated. The graph on the right represents the data predicted by our model. The next window will show this done again with some generated testing data. This process is then repeated for different sigma values. After eight windows, you should see a summary graph showing how sigma value corresponds with error.*

Original training data and predicted training data: 
<p align="center">
  <img width="1294" alt="Screen Shot 2023-08-28 at 9 32 05 PM" src="https://github.com/andy-x-li/Machine-Learning/assets/125074849/bbc94e52-85ed-4821-bc18-71e7493eaae0">

Original testing data and predicted testing data: 
<p align="center">
  <img width="1289" alt="Screen Shot 2023-08-28 at 9 33 46 PM" src="https://github.com/andy-x-li/Machine-Learning/assets/125074849/e5235f8e-ce29-40ac-932e-b6d734488f1c">
</p>

Error Summary Graph: 
<p align="center">
  <img width="791" alt="Screen Shot 2023-08-28 at 11 17 49 PM" src="https://github.com/andy-x-li/Machine-Learning/assets/125074849/87720178-5def-4783-b9b0-3a7e5c46e520">
</p>
