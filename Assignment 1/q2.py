import numpy as np
import sys
import collections

EPSILON = 1e-10

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

def _loss(x, y, theta):
  loss = 0
  for (i, x_i) in enumerate(x):
    mat_mult = np.dot(theta.T, x_i)
    loss += (y[i] - mat_mult) ** 2
  return loss / (2 * np.shape(x[:, 0])[0])

def _gradient(x, y, theta):    
  gradient = np.zeros_like(x[0])

  for (i, x_i) in enumerate(x):
    mat_mult = np.dot(theta.T, x_i)
    # Calculating the gradient
    gradient += (y[i] - mat_mult) * (-x_i)

  return gradient / np.shape(x[:, 0])[0]

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

  buffer = collections.deque([], 1000)

  diff = np.inf
  count = 0

  total_samples = np.shape(x[:, 0])[0]

  # # Plotting the 3-d mesh
  # x1 = np.linspace(-0.01, 0.01, 100)
  # y1 = np.linspace(-0.5, 1.5, 100)
  # X, Y = np.meshgrid(x1, y1)
  # Z = []

  # # Making Z list of thetas
  # for (i, x_i) in enumerate(X):
  #   Z.append([])
  #   for (j, x_j) in enumerate(x_i):
  #     Z[i].append(_loss(x, y, np.array([x_j, Y[i][j]])))
  # Z = np.array(Z)

  # # Plotting the mesh
  # ax = plt.axes(projection='3d')
  # ax.plot_wireframe(X, Y, Z, color='green', linewidths=0.5)


  # # Labels
  # ax.set_xlabel('Theta[1]')
  # ax.set_ylabel('Theta[0]')
  # ax.set_zlabel('Cost')

  while True:
    
    for i in range(0, total_samples, batch_size):
      
      x_temp = x[i : i + batch_size, :]
      y_temp = y[i : i + batch_size]

      # Gradient descent
      new_theta = old_theta - (LEARNING_RATE * _gradient(x_temp, y_temp, old_theta))

      buffer.append(_loss(x_temp, y_temp, new_theta))
      # Calculating new cost
      new_cost = np.average(buffer)
      print("Iteration {} => {}".format(count, new_cost))
      
      diff = abs(new_cost - old_cost)

      if diff < EPSILON:
        return new_theta
      # Updating theta
      old_theta = new_theta
      old_cost = new_cost

      count += 1
    # Plotting every 10th value
    # if count % 10 == 0:
    #   ax.scatter3D(new_theta[0], new_theta[1], new_cost)
    #   plt.pause(0.2)
    

  # plt.show()
  # return new_theta


if __name__ == "__main__":
  
  try:
    batch_size = int(sys.argv[1])
  except expression as identifier:
    batch_size = 100
  
  # sample_points()

  # Reading the data and modifying to desired form
  k = np.loadtxt('samples.txt')
  print(np.shape(k))
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
  print(_loss(x, y, theta))


