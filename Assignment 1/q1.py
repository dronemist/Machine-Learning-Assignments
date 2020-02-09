from mpl_toolkits import mplot3d
from numpy import linalg as lg
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import math

# Stopping gradient descent 
EPSILON = 1e-10

# Learning rate for gradient descent
LEARNING_RATE = 0.01

def plot_2D(x, y, theta):
  """
  Plotting hypothesis function on a plane
  """
  # Plotting actual data
  plt.scatter(x[:, 0], y, color= "blue",  
            marker= "o", s=10)
  prediction = np.array([np.dot(theta.T, x_i) for x_i in x])
  # plotting the hypothesis function
  plt.plot(x[:, 0], prediction, color = "green")

  # Assigning labels
  plt.xlabel('Normalised acidity')
  plt.ylabel('Density')

  plt.show()

def normalise(x):
  """
  Normalise x to have 0 mean and 1 variance
  """
  mean = np.mean(x, axis=0)
  variance = np.var(x, axis=0)
  x = ((x - mean)/ (variance ** 0.5))
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

def batch_gradient_descent(x, y, draw_3d = True):
  """
  Applies batch gradient descent to x and y
  return the parameters learnt 
  """
  new_theta = np.zeros_like(x[0])
  old_theta = new_theta
  # new_theta = np.array([1, 2])
  # old_theta = new_theta

  old_cost = 0
  new_cost = 0

  diff = np.inf
  count = 0

  # Plotting the 3-d mesh
  if draw_3d:
    x1 = np.linspace(-0.01, 0.01, 100)
    y1 = np.linspace(-0.5, 1.5, 100)
  else:
    x1 = np.linspace(-1, 1, 100)
    y1 = np.linspace(-0.5, 2.5, 100)  
  X, Y = np.meshgrid(x1, y1)
  Z = []

  # Making Z list of thetas
  for (i, x_i) in enumerate(X):
    Z.append([])
    for (j, x_j) in enumerate(x_i):
      Z[i].append(_loss(x, y, np.array([x_j, Y[i][j]])))
  Z = np.array(Z)

  # Plotting the mesh
  left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
  if draw_3d:
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, Z, color='green', linewidths=0.5)
  else:
    # Plotting the contours
    f2 = plt.figure(2)
    ax = f2.add_axes([left, bottom, width, height])
    cp = ax.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.clabel(cp, inline=1, fontsize=10)


  # Labels
  ax.set_xlabel('Theta[1]')
  ax.set_ylabel('Theta[0]')
  if draw_3d:
    ax.set_zlabel('Cost')

  while diff > EPSILON:
    
    # Gradient descent
    new_theta = old_theta - (LEARNING_RATE * _gradient(x, y, old_theta))

    # Calculating new cost
    new_cost = _loss(x, y, new_theta)
    diff = abs(new_cost - old_cost)

    # Updating theta
    old_theta = new_theta
    old_cost = new_cost

    # Plotting every 10th value
    if count % 10 == 0:
      if draw_3d:
        ax.scatter3D(new_theta[0], new_theta[1], new_cost)
      else: 
        ax.scatter(new_theta[0], new_theta[1], marker='o')  
      plt.pause(0.2)
    count += 1

  plt.show()
  return new_theta

if __name__ == "__main__":

  # If a 3d graph has to be plotted
  try:
    draw_3D_graphs = float(sys.argv[1])
  except:
    draw_3D_graphs = True

  # custom learning rate
  try:
    learning_rate = float(sys.argv[2])
  except:
    learning_rate = 0.01   

  # Reading csv file and adding 1 for the intercept term
  acidity = __add_dimension(normalise(read_csv_file('data/q1/linearX.csv')))
  density = read_csv_file('data/q1/linearY.csv')

  # Applying gradient descent
  LEARNING_RATE = learning_rate
  theta = batch_gradient_descent(acidity, density, draw_3D_graphs)

  # Displaying the learnt parameters and graphs
  print(_loss(acidity, density, theta))
  plot_2D(acidity, density, theta)
  print("The line is: {}x + {}".format(*theta))
  # for x in acidity:
  #   print(np.dot(theta.T, x))

  