import numpy as np
import matplotlib.pyplot as plt
import math

def normalise(x):
  """
  Normalise x to have 0 mean and 1 variance
  """
  mean = np.mean(x, axis=0)
  variance = np.var(x, axis=0)
  x = ((x - mean) / (variance ** 0.5))
  return x 

# P(Y = 1; theta) = phi
def phi(y):
  total_samples = np.shape(y)[0]
  count = 0
  for y_i in y:
    if y_i == 1:
      count += 1
  return count / total_samples    


# Taken alaska as y = 0
def mean(x, y, isCanada):

  numerator = np.zeros_like(x[0])
  denominator = 0

  for (i, x_i) in enumerate(x): 
    if y[i] == float(isCanada):
      numerator += x_i
      denominator += 1
  return numerator / denominator 

def covariance(x, y):
  total_parameters = np.shape(x[-1])[0]
  total_examples = np.shape(x[:, 0])[0]
  covariance = np.zeros((total_parameters, total_parameters))
  mean_array = np.array([mean(x, y, 0), mean(x, y, 1)])

  for (i, x_i) in enumerate(x):
    temp_x = np.reshape(x_i - mean_array[int(y[i])], (total_parameters, 1))
    covariance += np.dot(temp_x, temp_x.T)
  
  return covariance / total_examples

def general_covariance(x, y, isCanada):
  total_parameters = np.shape(x[-1])[0]
  denominator = 0
  numerator = np.zeros((total_parameters, total_parameters))
  mean_yi = mean(x, y, isCanada)

  for(i, x_i) in enumerate(x):
    if y[i] == float(isCanada):
      matrix = np.reshape(x_i - mean_yi, (total_parameters, 1))
      denominator += 1
      numerator += np.dot(matrix, matrix.T)
  return numerator / denominator

def plot_2D(x, y):
  """
  Plotting hypothesis function on a plane
  """
  oneAlaska = False
  oneCanada = False
  # Plotting actual data
  for (i, y_i) in enumerate(y):
    if(y_i == 1):
      if not oneCanada:
        plt.plot(x[i, 0], x[i, 1], color= "blue",  
            marker= "x", mew=1, ms=5, label="Canada")
        oneCanada = True    
      else:
        plt.plot(x[i, 0], x[i, 1], color= "blue",  
            marker= "x", mew=1, ms=5)      
    else:
      if not oneAlaska:
        plt.plot(x[i, 0], x[i, 1], color= "red",  
            marker= "+", mew=1, ms=5, label="Alaska")   
        oneAlaska = True    
      else:
        plt.plot(x[i, 0], x[i, 1], color= "red",  
            marker= "+", mew=1, ms=5)  


  # Plotting the linear boundary line
  # Required terms
  x_linear = np.linspace(-3, 3, 100)
  sigma_inverse = np.linalg.inv(covariance(x, y))
  mu0 = mean(x, y, 0)
  mu1 = mean(x, y, 1)
  phi_y = phi(y)
  sigma_inverse_mu_1 = np.dot(sigma_inverse, mu1)
  sigma_inverse_mu_0 = np.dot(sigma_inverse, mu0)
  K_linear = sigma_inverse_mu_1 - sigma_inverse_mu_0
  constant_linear = ((np.dot(mu1.T, sigma_inverse_mu_1) - np.dot(mu0.T, sigma_inverse_mu_0)) / 2)
  + math.log((1 - phi_y) / phi_y)

  # Plotting the quadratic boundary
  sigma1 = general_covariance(x, y, 1)
  sigma0 = general_covariance(x, y, 0)
  sigma1_inverse = np.linalg.inv(sigma1)
  sigma0_inverse = np.linalg.inv(sigma0)
  sigma1_inverse_mu_1 = np.dot(sigma1_inverse, mu1)
  sigma0_inverse_mu_0 = np.dot(sigma0_inverse, mu0)
  K1_quadratic = sigma0_inverse_mu_0 - sigma1_inverse_mu_1
  K2_quadratic = sigma1_inverse - sigma0_inverse
  constant_quadratic = - ((np.dot(mu1.T, sigma1_inverse_mu_1) - np.dot(mu0.T, sigma0_inverse_mu_0)) / 2)
  + math.log((1 - phi_y) / phi_y) + (math.log(np.linalg.det(sigma0) / np.linalg.det(sigma1)) / 2)

  a = K2_quadratic[0][0]
  b = K2_quadratic[0][1] + K2_quadratic[1][0]
  c = K2_quadratic[1][1]
  d = K1_quadratic[0]
  e = K1_quadratic[1]
  f = -constant_quadratic

  # Equation of the line
  y_linear = -x_linear * (K_linear[0] / K_linear[1]) + constant_linear / K_linear[1]
  print("Equation of linear boundary is: {}x + {}".format(-K_linear[0]/K_linear[1], constant_linear/K_linear[1]))
  plt.plot(x_linear, y_linear)

  # Equation of quadratic boundary
  x_quadratic = np.linspace(-3, 3, 100)
  y_qudratic = np.linspace(-2, 4, 100)
  x_quadratic, y_qudratic = np.meshgrid(x_quadratic, y_qudratic)
  plt.contour(x_quadratic, y_qudratic,(a*x_quadratic**2 + b*x_quadratic*y_qudratic + c*y_qudratic**2 + d*x_quadratic + e*y_qudratic + f), [0], colors='green')

  # Assigning labels
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.legend()
  plt.title('GDA')
  plt.show()



if __name__ == "__main__":
  # Loading and preparing x
  x = normalise(np.loadtxt('./data/q4/q4x.dat'))

  # Loading and preparing y
  y = np.loadtxt('./data/q4/q4y.dat', dtype=str)
  y[y == "Alaska"] = float(0)
  y[y == "Canada"] = float(1) 
  y = np.array(y, dtype=float)

  print("Sigma 1 = Sigma 0 =>")
  print("Phi: {}".format(phi(y)))
  print("Mu0: {}".format(mean(x, y, 0))) 
  print("Mu1: {}".format(mean(x, y, 1)))
  print("Sigma: {}".format(covariance(x, y)))  
  plot_2D(x, y)

  print("Sigma 1 != Sigma 0 =>")
  print("Phi: {}".format(phi(y)))
  print("Mu0: {}".format(mean(x, y, 0))) 
  print("Mu1: {}".format(mean(x, y, 1)))
  print("Sigma1: {}".format(general_covariance(x, y, 1)))
  print("Sigma0: {}".format(general_covariance(x, y, 0)))
