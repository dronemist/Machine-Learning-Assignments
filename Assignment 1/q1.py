from numpy import linalg as lg
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# Stopping gradient descent 
EPSILON = 0.00000000001

# Learning rate for gradient descent
LEARNING_RATE = 0.01

def plot_2D(x, y, theta):
  # Plotting actual data
  plt.scatter(x[:, 0], y, color= "blue",  
            marker= "o", s=10)
  prediction = np.array([np.dot(theta.T, x_i) for x_i in x])
  # plotting the line
  plt.plot(x[:, 0], prediction, color = "green")

  # Assigning labels
  plt.xlabel('Normalised acidity')
  plt.ylabel('Normalised density')
  
  plt.show()

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

  x = ((x - mean) / variance)
  return x 

def read_csv_file(file_name):
  """
  Read Xi and Yi values
  """
  list = []
  with open(file_name, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      list.append(row)

  # Converting to float
  new_list = [[float(y) for y in x] for x in list]
  return np.array(new_list)

# Adding dimension for the intercept term
def __add_dimension(x):
  temp = np.ones_like(x)
  return np.append(x, temp, 1)

def _loss(x, y, theta):
  loss = 0
  for (i, x_i) in enumerate(x):
    mat_mult = np.dot(theta.T, x_i)
    loss += (y[i][0] - mat_mult) ** 2
  return loss / (2 * np.shape(x[:, 0])[0])

def _gradient(x, y, theta):    
  gradient = np.zeros_like(x[0])

  for (i, x_i) in enumerate(x):
    mat_mult = np.dot(theta.T, x_i)
    # Calculating the gradient
    gradient += (y[i][0] - mat_mult) * (-x_i)

  return gradient / np.shape(x[:, 0])[0]

def batch_gradient_descent(x, y):
  """
  Applies batch gradient descent to x and y
  return the parameters learnt 
  """
  new_theta = np.zeros_like(x[0])
  old_theta = np.zeros_like(x[0])

  old_cost = 0
  new_cost = 0

  diff = np.inf

  while diff > EPSILON:
    
    # Gradient descent
    new_theta = old_theta - (LEARNING_RATE * _gradient(x, y, old_theta))

    # Calculating new cost
    new_cost = _loss(x, y, new_theta)
    diff = abs(new_cost - old_cost)

    # Updating theta
    old_theta = new_theta
    old_cost = new_cost

  return new_theta

if __name__ == "__main__":

  # Reading csv file and adding 1 for the intercept term
  acidity = __add_dimension(normalise(read_csv_file('data/q1/linearX.csv')))

  density = read_csv_file('data/q1/linearY.csv')

  # Applying gradient descent
  theta = batch_gradient_descent(acidity, density)
  print(acidity)
  print(_loss(acidity, density, theta))
  plot_2D(acidity, density, theta)
  # for x in acidity:
  #   print(np.dot(theta.T, x))

  