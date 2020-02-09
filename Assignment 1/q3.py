from numpy import linalg as lg
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

EPSILON = 1e-10

def plot_2D(x, y, theta):
  """
  Plotting hypothesis function on a plane
  """
  # Plotting actual data
  for (i, y_i) in enumerate(y):
    if(y_i == 1):
      plt.plot(x[i, 1], x[i, 2], color= "blue",  
            marker= "x", mew=1, ms=5)
    else:
      plt.plot(x[i, 1], x[i, 2], color= "red",  
            marker= "+", mew=1, ms=5)        

  # Plotting logistic line
  y_temp = - (theta[1] / theta[2]) * x[:, 1] - (theta[0]/ theta[2])
  plt.plot(x[:, 1], y_temp)

  # Assigning labels
  plt.xlabel('x1')
  plt.ylabel('x2')

  plt.show()

def normalise(x):
  """
  Normalise x to have 0 mean and 1 variance
  """
  mean = np.mean(x, axis=0)
  variance = np.var(x, axis=0)
  x = ((x - mean)/ (variance ** 0.5))
  return x 

def sigmoid(x, theta):
  mat_mult = np.dot(theta.T, x)
  return (1 / (1 + np.exp(-mat_mult)))

def _gradient_logistic(x, y, theta):

  total_samples = np.shape(x[:, 0])[0]
  total_parameters = np.shape(x[-1])[0]
  gradient = np.zeros((total_parameters, 1))

  for (i, x_i) in enumerate(x):
    gradient += -1 * (y[i] - sigmoid(x_i, theta)) * np.reshape(x_i, (total_parameters, 1))
  
  return gradient / total_samples

def _hessian_logistic(x, y, theta):
  total_parameters = np.shape(x[-1])[0]
  hessian = np.zeros((total_parameters, total_parameters))
  
  for x_i in x:
    x_temp = np.reshape(x_i, (total_parameters, 1))
    sigma = sigmoid(x_i, theta)
    hessian += (sigma * (1 - sigma) * x_temp * (x_temp.T))
  return hessian

def _loss_logistic(x, y, theta):
  loss = 0
  
  for (i, x_i) in enumerate(x):
    sigma = sigmoid(x_i, theta)
    loss += (y[i] * math.log(sigma) + (1 - y[i]) * math.log(1 - sigma))
  return loss

def newton_method(x, y):
  """
  Applies batch gradient descent to x and y
  return the parameters learnt 
  """
  total_parameters = np.shape(x[-1])[0]
  new_theta = np.zeros((total_parameters, 1))
  old_theta = new_theta

  old_cost = 0
  new_cost = 0
  diff = np.inf
  count = 0

  while diff > EPSILON:
    
    # Gradient descent
    hessian = _hessian_logistic(x, y, old_theta)
    gradient = _gradient_logistic(x, y, old_theta)
    new_theta = old_theta  - np.dot(lg.inv(hessian), gradient)
    # Calculating new cost  
    # new_cost = _loss_logistic(x, y, new_theta)
    diff = abs(lg.norm(new_theta - old_theta))

    # Updating theta
    old_theta = new_theta
    # old_cost = new_cost

  return new_theta

if __name__ == "__main__":
  x = np.loadtxt('./data/q3/logisticX.csv', delimiter=',')
  x = normalise(x)
  print(np.max(x[:, 0]))
  x1 = np.ones_like(x[:, 0])
  x = np.column_stack((x1, x))

  y = np.loadtxt('./data/q3/logisticY.csv')
  theta = newton_method(x, y)
  print("Theta is: {}".format(theta))
  print("Logistic loss is: {}".format(_loss_logistic(x, y, theta)))
  plot_2D(x, y, theta)

  print("line is: {}x + {}".format(-(theta[1]/theta[2])[0], -(theta[0]/theta[2])[0]))
