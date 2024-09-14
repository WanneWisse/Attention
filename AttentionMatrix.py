import tensorflow as tf
import ModelLogic
import numpy as np
import matplotlib.pyplot as plt
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#take an example
x_test = x_test[1]
#prepare for model
x_test_normalised = np.array([x_test.astype('float32') / 255.0])

#load model
new_model = ModelLogic.create_model()
new_model.load_weights('WithAttention.h5')

#predict and get attention_matrix of QK
scores, attention_matrix = new_model.predict(x_test_normalised)
attention_matrix= attention_matrix[0]

#average over rows or columns (axis= 1 or 0) and shape to image from 784-> 28x28 
sum_attention_pixels = np.sum(attention_matrix, axis=1) / len(attention_matrix)
sum_attention_pixels = sum_attention_pixels.reshape((28,28))

#plot attentionmap
fig, ax = plt.subplots()
ax.matshow(sum_attention_pixels,  cmap='Blues')
plt.show()