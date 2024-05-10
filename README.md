# Exp:05 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
## Program:
```
Developed by: KANISHKA V S
RegisterNumber: 212222230061 

```
```py
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
### Array Value of x
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/a7362030-fda6-431f-aba5-269ed202e451)

### Array Value of y
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/9a91ca9e-0601-458c-93f3-5b0fe92c2aa9)

### Exam 1 - score graph
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/86b97db5-5ceb-40bc-ab77-e33248d5fd38)

#### Sigmoid function graph
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/6a632902-26ce-4881-8c17-4d2b1921bd4b)

### X_train_grad value
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/533c332d-ff4b-4026-9fe2-bde7b9f5f13d)

### Y_train_grad value
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/0603292a-9181-47da-a4e5-d60e6f0532e6)

### Print res.x
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/08687611-c9df-4db6-9479-029c5c01bc77)

### Decision boundary - graph for exam score
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/62ac8d2b-27c9-4c02-913b-6c7eb7836f08)

### Proability value
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/1e487849-ee00-4f78-9355-805dd5def7d3)

### Prediction value of mean
![image](https://github.com/kanishka2305/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497357/7f4267b7-76bd-4881-9de9-39279b0e2681)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

