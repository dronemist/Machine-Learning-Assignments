import numpy as np

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
  np.savetxt('samples.txt', y)

if __name__ == "__main__":
  # sample_points()


