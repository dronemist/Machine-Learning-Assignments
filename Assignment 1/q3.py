from numpy import linalg as lg
import numpy as np
import math
import csv
import matplotlib as plt

EPSILON = 1e-10

def normalise(x):
  """
  Normalise x to have 0 mean and 1 variance
  """
  total_samples = np.shape(x[:, 0])[0]
  mean = 0
  variance = 0

  for x_i in x:
    mean += x_i 
  mean /= total_samples

  for x_i in x:
    variance += (x_i - mean) ** 2
  variance /= total_samples

  # Converting to numpy array
  mean = np.array(mean)
  variance = np.array(variance)

  x = ((x - mean)/ variance)
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
    new_cost = _loss_logistic(x, y, new_theta)
    diff = abs(new_cost - old_cost)
    
    # Updating theta
    old_theta = new_theta
    old_cost = new_cost

  return new_theta

if __name__ == "__main__":
  x = np.loadtxt('./data/q3/logisticX.csv', delimiter=',')
  x1 = np.ones_like(x[:, 0])
  x = np.column_stack((x1, x))
  y = np.loadtxt('./data/q3/logisticY.csv')
  theta = newton_method(x, y)
  print(theta)
  print(_loss_logistic(x, y, theta))