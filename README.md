# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: M Sanjay
RegisterNumber:  212222240090
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=",")
X=data[:,[0,1]]
Y=data[:,2]

X[:5]

Y[:5]

plt.figure()
plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label="Admitted")
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-Y)/X.shape[0]
  return J,grad
  
X_train=np.hstack((np.ones((X.shape[0], 1)), X))
theta= np.array([0,0,0])
J, grad= costFunction(theta, X_train, Y)
print(J)
print(grad)  

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, Y)
print(J)
print(grad)

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad
    
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun = cost, x0 = theta, args = (X_train, Y), method = "Newton-CG", jac = gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,Y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    Y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[Y == 1][:, 0], X[Y ==1][:, 1], label="Admitted")
    plt.scatter(X[Y == 0][:, 0], X[Y ==0][:, 1], label=" Not Admitted")
    plt.contour(xx,yy,Y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
    
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
prob

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X)==Y)
*/
```

## Output:
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/703d0e5e-29c7-4cf6-8bef-f5258713ed3d)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/6831b713-d737-4684-8d9d-cfc8b2c138b7)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/dbf628da-173f-4023-afe3-c001ba3169fd)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/89a3ad76-bae8-46d3-b9c3-d6ae8e6c4323)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/15d2665c-722b-4a6e-9c82-01e22d8c8c39)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/1ad76537-25e7-4145-94cc-cf71c5ca9f8a)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/80e39a8e-515d-426e-b994-ca4744d6bb58)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/aa3547ef-9220-4862-8164-076417c89a51)
![image](https://github.com/Sanjay22006832/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119830477/d25c72c2-b6db-412c-bbab-87266c5f10d6)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

