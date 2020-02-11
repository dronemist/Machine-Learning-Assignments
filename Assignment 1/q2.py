import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
import math
import collections

EPSILON = 1e-3

LEARNING_RATE = 0.001

def sample_points():
  """
  Sampling one million points
  """
  NUMBER_OF_SAMPLES = 1000000
  SAMPLE_THETA = np.array([3, 1, 2])
  x1 = np.random.normal(3, 2, NUMBER_OF_SAMPLES)
  x2 = np.random.normal(-1, 2, NUMBER_OF_SAMPLES)
  noise = np.random.normal(0, 2**0.5, NUMBER_OF_SAMPLES)
  y = []
  for i in range(0, NUMBER_OF_SAMPLES):
    y.append(SAMPLE_THETA[0] * 1 + SAMPLE_THETA[1] * x1[i] + SAMPLE_THETA[2] * x2[i] + noise[i])
  np.savetxt('samples.txt', np.c_[x1, x2, y])

def _loss(x, y, theta):
  total_parameters = np.shape(x[-1])[0]
  total_samples = np.shape(x[:, 0])[0]
  theta = np.reshape(theta, (total_parameters, 1))
  x = x.T
  y = np.reshape(y, (1, total_samples))
  loss = y - np.matmul(theta.T, x)
  loss = np.sum(np.square(loss))
  return loss / (2 * total_samples)

def _gradient(x, y, theta):    
  total_parameters = np.shape(x[-1])[0]
  total_samples = np.shape(x[:, 0])[0]
  theta = np.reshape(theta, (total_parameters, 1))
  y = np.reshape(y, (1, total_samples))
  gradient = -np.matmul((y - np.matmul(theta.T, x.T)), x)
  answer = gradient[0] / total_samples
  return answer

def stochastic_gradient_descent(x, y, batch_size):
  """
  Applies batch gradient descent to x and y
  return the parameters learnt 
  """
  new_theta = np.zeros_like(x[0])
  old_theta = new_theta

  old_cost = 0
  new_cost = 0

  diff = np.inf
  count = 0

  total_samples = np.shape(x[:, 0])[0]
  CHECK_N = 2000
  # # Plotting the mesh
  # ax = plt.axes(projection='3d')

  # # Labels
  # ax.set_xlabel('Theta[0]')
  # ax.set_ylabel('Theta[1]')
  # ax.set_zlabel('Theta[2]')

  while True:
    for i in range(0, total_samples, batch_size):
      
      x_curr = x[i : i + batch_size, :]		
      y_curr = y[i : i + batch_size]
      
      # Gradient descent
      new_theta = old_theta - (LEARNING_RATE * _gradient(x_curr, y_curr, old_theta))
      new_cost += (np.dot(new_theta, new_theta) ** 0.5)
      # new_cost += _loss(x_curr, y_curr, new_theta)
      # new_cost /= 2
      
      if count % int(total_samples / batch_size) == 0:
        print("Iteration {} => {}".format(count, new_theta))
      
      count += 1

      if count % CHECK_N == 0:
        new_cost /= CHECK_N
        
        diff = abs(new_cost - old_cost)
        old_cost = new_cost
        new_cost = 0
        if diff < EPSILON or count > 120000:
          print("Number of updates: {}".format(count))
          plt.show()
          return new_theta

      # Plotting every 100th value
      # if count % 1 == 0:
      #   ax.scatter3D(new_theta[0], new_theta[1], new_theta[2])
      #   plt.pause(0.2)
      
      # Updating theta
      old_theta = new_theta
      


if __name__ == "__main__":
  
  try:
    batch_size = int(sys.argv[1])
  except:
    batch_size = 100

  try:
    train = int(sys.argv[2])
  except:
    train = 0
  
  # sample_points()
  if train == 1:
    # Reading the data and modifying to desired form
    k = np.loadtxt('samples.txt')
    # Shuffling data
    np.random.shuffle(k)
    x1 = k[:, 0]
    x2 = k[:, 1]
    y = k[:, 2]
    x0 = np.ones_like(x1)
    x12 = np.column_stack((x1, x2))
    x = np.column_stack((x0, x12))
    
    # Applying SGD 
    theta = stochastic_gradient_descent(x, y, batch_size)
    # theta = np.zeros_like(x[0])
    print(theta)
    print("Loss on training data set: {}".format(_loss(x, y, theta)))
    np.savetxt(str(batch_size) + '_theta_values.txt', theta)

  theta = np.loadtxt(str(batch_size) + '_theta_values.txt')
  # Testing on given data
  k_test = np.loadtxt('./data/q2/q2test.csv', delimiter=',', skiprows=1)
  x1 = k_test[:, 0]
  x2 = k_test[:, 1]
  y = k_test[:, 2]
  x0 = np.ones_like(x1)
  x12 = np.column_stack((x1, x2))
  x = np.column_stack((x0, x12))
  print("Loss on given dataset: {}".format(_loss(x, y, theta)))
