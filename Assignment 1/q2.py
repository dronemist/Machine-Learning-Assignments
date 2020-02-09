import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys
import collections

EPSILON = 1e-2

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
    y.append(SAMPLE_THETA[0] * 1 + SAMPLE_THETA[1] * x1[i] + SAMPLE_THETA[2] * x2[i])
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
  # new_theta = np.array([1, 2])
  # old_theta = new_theta

  old_cost = 0
  new_cost = 0

  diff = np.inf
  count = 0

  total_samples = np.shape(x[:, 0])[0]
  buffer = collections.deque([], 1000)

  # Plotting the mesh
  ax = plt.axes(projection='3d')

  # Labels
  ax.set_xlabel('Theta[0]')
  ax.set_ylabel('Theta[1]')
  ax.set_zlabel('Theta[2]')

  while True:
    for i in range(0, total_samples, batch_size):
      
      # x_temp = x[i : i + batch_size, :]
      # y_temp = y[i : i + batch_size]
      # Gradient descent
      new_theta = old_theta - (LEARNING_RATE * _gradient(x, y, old_theta, i, i + batch_size))

      buffer.append(_loss(x, y, new_theta, i, i + batch_size))
      # Calculating new cost
      new_cost = np.average(buffer)
      if count % (total_samples / batch_size) == 0:
        print("Iteration {} => {}".format(count, new_cost))
      
      diff = abs(new_cost - old_cost)

      if diff < EPSILON:
        print("Number of updates: {}".format(count))
        plt.show()
        return new_theta
      # Updating theta
      old_theta = new_theta
      old_cost = new_cost

      # Plotting every 100th value
      if count % 1 == 0:
        ax.scatter3D(new_theta[0], new_theta[1], new_theta[2])
        plt.pause(0.2)
    
      count += 1    
  return new_theta


if __name__ == "__main__":
  
  try:
    batch_size = int(sys.argv[1])
  except:
    batch_size = 100
  
  # sample_points()

  # Reading the data and modifying to desired form
  k = np.loadtxt('samples.txt')
  # Shuffling data
  # np.random.shuffle(k)
  x1 = k[:, 0]
  x2 = k[:, 1]
  y = k[:, 2]
  x0 = np.ones_like(x1)
  x12 = np.column_stack((x1, x2))
  x = np.column_stack((x0, x12))
  
  # Applying SGD 
  # theta = stochastic_gradient_descent(x, y, batch_size)
  # print(theta)
  # print(_loss(x, y, theta, 0, np.shape(x[:, 0])[0]))


  # Testing on given data
  k_test = np.loadtxt('./data/q2/q2test.csv', delimiter=',', skiprows=1)
  print(k_test)
  x1 = k_test[:, 0]
  x2 = k_test[:, 1]
  y = k_test[:, 2]
  x0 = np.ones_like(x1)
  x12 = np.column_stack((x1, x2))
  x = np.column_stack((x0, x12))
  print(_loss(x, y, theta, 0, np.shape(x[:, 0])[0]))
