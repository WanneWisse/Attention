import tensorflow as tf
import matplotlib.pyplot as plt
import ModelLogic


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [0, 1] range
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Print shapes of the datasets
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Convert labels to one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Batch size
batch_size = 64

# Create a TensorFlow Dataset object from the training and test data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Shuffle, batch the training dataset and prefetch to optimise image processing and training 
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Batch the test dataset (no need to shuffle test data) and prefetch to optimise image processing and training 
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# Create and summarize the model
model = ModelLogic.create_model()
model.summary()

#set some epochs and fit the model
epochs=2
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=epochs
)

#get the loss of the model on train and validation
loss = history.history['loss']
val_loss = history.history['val_loss']

#plot loss in figure
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#save weights of model, saving complete model in .keras is harder because of attention class
model.save('WithAttention.h5')  





