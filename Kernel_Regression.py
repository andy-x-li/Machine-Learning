from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy.io as sio
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys

def cal_kernel(sigma):
  """
  cal_kernel takes in a sigma value and solves for K and alpha from the training data (X). It then 
  estimates the predicted label (y2) using K and alpha. Afterwards, it estimates the error between 
  the predicted label (y2) and the training data (y). cal_kernel can then, calculate K with the test 
  data and estimate the test data's predicted label (ytest2). The error between the predicted label 
  (ytest2) and the test data (ytest) is then calculated.  
  """
  print("sigma^2:",sigma)
  n = 200
  p = 2
  X = 2*(np.random.rand(n,p)-.5)  # generate random numbers from -1 to 1
  y = np.sign(X[:,1]-(X[:,0]**2/2+np.sin(X[:,0]*7)/2))  # y is either 1 or -1
  #print("X.shape:",X.shape)
  #print("y.shape:",y.shape)

  # Train a kernel ridge regression using a Gaussian kernel
  lam = 1

  norms2 = (np.array(la.norm(X,axis=1)).T)**2 # squared norm of each training sample
  innerProds = X@X.T
  #print("norms2.shape",norms2.shape)

  # squared distances between each pair of training samples
  dist2 = np.matrix(norms2).T@np.ones([1,n]) + np.ones([n,1])@np.matrix(norms2) - 2*innerProds 

  K = np.exp(-1*dist2/(2*sigma))  # K's dimension is 200x200
  #print("K.shape",K.shape)
  alpha = la.inv(K+lam*np.identity(n))@y # alpha's dimension is 1x200
  #print("alpha.shape:", alpha.shape)
  yhat = K@alpha.T  # yhat's dimension is 200x1
  #print("yhat.shape:", yhat.shape)
  y2 = np.array(np.sign(yhat))

  # calculate error for training sample data
  err1 = np.zeros(1)
  for k in range(y.shape[0]):
   if(y2[k] != y[k]):
     err1 = err1 + 1
  err1 = err1/y.shape[0]
  print("error for training data:", err1)

  ntest = 2000
  Xtest = 2*(np.random.rand(ntest,p)-0.5)
  ytest = np.sign(Xtest[:,1]-(Xtest[:,0]**2/2+np.sin(Xtest[:,0]*7)/2))  
  norms2_test = (np.array(la.norm(Xtest,axis=1)).T)**2 # squared norm of each training sample
  innerProds_test = Xtest@X.T
  #print("norms2_test.shape",norms2_test.shape)
  dist2_test = np.matrix(norms2_test).T@np.ones([1,n]) + np.ones([ntest,1])@np.matrix(norms2) - 2*innerProds_test
  K_test = np.exp(-1*dist2_test/(2*sigma))  # K_test's dimension is 2000x200
  #print("K_test.shape:", K_test.shape)
  ytesthat = K_test@alpha.T   # ytesthat's dimension is 2000x1
  ytest2 = np.array(np.sign(ytesthat))  
  #print("ytesthat.shape",ytesthat.shape)
  #print("Xtest.shape",Xtest.shape)

  # plot
  ax1 = plt.subplot(121)
  ax1.scatter(X[:,0],X[:,1],50, c=y)
  ax1.set_xlabel('feature 1')
  ax1.set_ylabel('feature 2')
  ax1.set_title('Training data colored by label (sigma^2:' +str(sigma) + ')')

  ax2 = plt.subplot(122)
  ax2.scatter(X[:,0],X[:,1],50,c=y2)
  ax2.set_xlabel('feature 1')
  ax2.set_ylabel('feature 2')
  ax2.set_title('Training data colored by PREDICTED label (sigma^2:'+str(sigma) + ')')
  plt.show()

  ax3 = plt.subplot(121)
  ax3.scatter(Xtest[:,0],Xtest[:,1],50,c=ytest)
  ax3.set_xlabel('feature 1')
  ax3.set_ylabel('feature 2')
  ax3.set_title('Test data colored by label (sigma^2:'+str(sigma)+')')

  ax4 = plt.subplot(122)
  ax4.scatter(Xtest[:,0],Xtest[:,1],50,c=ytest2)
  ax4.set_xlabel('feature 1')
  ax4.set_ylabel('feature 2')
  ax4.set_title('Test data colored by PREDICTED label (sigma^2:' +str(sigma)+')')
  plt.show()

  # calculate error for test data
  err2 = np.zeros(1)
  for k in range(ytest.shape[0]):
   if(ytest2[k] != ytest[k]):
     err2 = err2 + 1
  err2 = err2/ytest.shape[0]
  print("error for test data:", err2)

  errors = np.zeros((2))
  errors[0] = err1
  errors[1] = err2
  return errors

# main code
serr = np.array([[0.025],[0.05],[0.1],[0.5]])
serrfin = np.zeros((2,4))

plt.rcParams['figure.figsize'] = [13, 6]
serrfin[:,0] = cal_kernel(0.025)
serrfin[:,1] = cal_kernel(0.05)
serrfin[:,2] = cal_kernel(0.1)
serrfin[:,3] = cal_kernel(0.5)

print(serrfin) 

f,(ax1,ax2) = plt.subplots(2, figsize=(8, 6))
ax1.set_ylabel('Error from Training Data')
ax1.plot(serr,serrfin[0,:])
ax2.set_ylabel('Error from Test Data')
ax2.plot(serr,serrfin[1,:])
plt.xlabel("sigma^2")
plt.show()
