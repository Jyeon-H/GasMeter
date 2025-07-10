import numpy as np

def load_data(num_classes=10, samples=75, image_shape=(28,28):
  total = num_classes * samples
  x = np.random.rand(total * image_shape)
  y = np.array([i // samples for i in range(total)])
  return x, y
