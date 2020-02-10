import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
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

def _loss(x, y, theta, start_index, end_index):
  loss = 0
  for i in range(start_index, end_index):
    mat_mult = np.dot(theta.T, x[i])
    loss += (y[i] - mat_mult) ** 2
  return loss / (2 * (end_index - start_index))

def _gradient(x, y, theta, start_index, end_index):    
  gradient = np.zeros_like(x[0])

  for i in range(start_index, end_index):
    mat_mult = np.dot(theta.T, x[i])
    # Calculating the gradient
    gradient += (y[i] - mat_mult) * (-x[i])

  return gradient / (end_index - start_index)

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
  threshold = min(4000, int(2 * total_samples / batch_size))
  if batch_size == 1:
    threshold = 20000 
  buffer = np.array([])

  # # Plotting the mesh
  # ax = plt.axes(projection='3d')

  # # Labels
  # ax.set_xlabel('Theta[0]')
  # ax.set_ylabel('Theta[1]')
  # ax.set_zlabel('Theta[2]')

  while True:
    for i in range(0, total_samples, batch_size):
      
      # Gradient descent
      new_theta = old_theta - (LEARNING_RATE * _gradient(x, y, old_theta, i, i + batch_size))

      length = len(buffer)
      if length == threshold:
        buffer = buffer[1 : length]
      buffer = np.append(buffer, _loss(x, y, new_theta, i, i + batch_size))
      
      if count % int(total_samples / batch_size) == 0:
        print("Iteration {} => {}".format(count, new_theta))

      # If buffer is not full don't check convergence
      if count > threshold: 
        # Calculating new cost
        new_cost = np.mean(buffer[int(length / 2) : threshold])
        old_cost = np.mean(buffer[0 : int(length / 2)])
        
        diff = abs(new_cost - old_cost)

        if diff < EPSILON:
          print("Number of updates: {}".format(count))
          plt.show()
          return new_theta

      # Plotting every 100th value
      # if count % 100 == 0:
      #   ax.scatter3D(new_theta[0], new_theta[1], new_theta[2])
      #   plt.pause(0.2)
      
      # Updating theta
      old_theta = new_theta

      count += 1    
  return new_theta


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
    print(theta)
    print("Loss on training data set: {}".format(_loss(x, y, theta, 0, np.shape(x[:, 0])[0])))
    np.savetxt(str(batch_size) + '_theta_values.txt', theta)

  theta = np.loadtxt(str(batch_size) + '_theta_values.txt')
  # theta = np.array([3, 1, 2])
  # Testing on given data
  k_test = np.loadtxt('./data/q2/q2test.csv', delimiter=',', skiprows=1)
  x1 = k_test[:, 0]
  x2 = k_test[:, 1]
  y = k_test[:, 2]
  x0 = np.ones_like(x1)
  x12 = np.column_stack((x1, x2))
  x = np.column_stack((x0, x12))
  print("Loss on given dataset: {}".format(_loss(x, y, theta, 0, np.shape(x[:, 0])[0])))
