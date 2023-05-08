import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model
input_shape = (382, 256)
num_nodes = 16


class STGCN(tf.keras.layers.Layer):
    def __init__(self, num_nodes, adj_mx, filters):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.adj_mx = adj_mx
        self.filters = filters
        self.conv_layers = []
        self.pooling_layers = []
        for i, f in enumerate(filters):
            self.conv_layers.append(Conv2D(f, (1, 9), activation='relu', padding='same'))
            self.pooling_layers.append(MaxPooling2D(pool_size=(1, 2)))
    
    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        for i, conv_layer in enumerate(self.conv_layers):
            x = tf.transpose(x, perm=[0, 1, 3, 2])
            x = tf.reshape(x, [-1, self.num_nodes, x.shape[2], x.shape[3]])
            x = tf.matmul(self.adj_mx, x)
            x = tf.transpose(x, perm=[0, 1, 3, 2])
            x = conv_layer(x)
            x = self.pooling_layers[i](x)
        x = Flatten()(x)
        return x

class STCN(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(STCN, self).__init__()
        self.filters = filters
        self.conv_layers = []
        self.pooling_layers = []
        for f in filters:
            self.conv_layers.append(Conv2D(f, (3, 3), activation='relu', padding='same'))
            self.pooling_layers.append(MaxPooling2D(pool_size=(2, 2)))
    
    def call(self, inputs):
        x = inputs
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.pooling_layers[i](x)
        x = Flatten()(x)
        return x

def create_model(input_shape, filters):
    inputs = Input(shape=input_shape)
    x = tf.expand_dims(inputs, axis=-1)
    x = STCN(filters=filters)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(101, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build Spatial Temporal Graph Convolution Network

model = create_model(input_shape=input_shape, filters=[32, 64, 128])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print("Model Structure of STGCN ")
print(model.summary())