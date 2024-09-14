import tensorflow as tf

#create attentionclass which also output attention matrix per prediction
class AttentionWithScores(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionWithScores, self).__init__(**kwargs)
        #use standard attention_layer of keras
        self.attention_layer = tf.keras.layers.Attention(use_scale=True)

    #when the layer is called
    def call(self, inputs):
        #the input is just 3 times the image flattend
        query, key, value = inputs
        #seperately multiply them with the query, key and value matrix (hidden in tensorflow implementation) and compute the attention output.
        attention_output, attention_scores = self.attention_layer([query, key, value], return_attention_scores=True)
        #next to the output which is attention_scores * V, we also want the attention_scores seperatly to inspect the behaviour of the network
        return attention_output, attention_scores


# Use Functional API because of attention layer
def create_model():
    
    # Input layers
    input_layer = tf.keras.Input(shape=(28, 28, 1))  # MNIST images

    # Flatten the input to make it compatible with attention layer
    flattened = tf.keras.layers.Flatten()(input_layer)  

    # Reshape for attention layer, add 1 extra dimension
    reshaped = tf.keras.layers.Reshape((784, 1))(flattened)

    # Define the attention layer
    attention_layer = AttentionWithScores()
    attention_output, attention_scores = attention_layer([reshaped, reshaped, reshaped])

    # Flatten the attention output and add a feedforward layer
    flattened_attention_output = tf.keras.layers.Flatten()(attention_output)
    dense_output = tf.keras.layers.Dense(128, activation='relu')(flattened_attention_output)
    output = tf.keras.layers.Dense(10, activation='softmax')(dense_output)

    # Build the model
    model = tf.keras.Model(inputs=input_layer, outputs=[output, attention_scores])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy', None],  # Only the first output has a loss
        metrics=['accuracy', None] # Only the first output has a metric
    )
    
    return model