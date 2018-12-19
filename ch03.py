from keras import layers

layer = layers.Dense(32, input_shape=(784,)) # A dense layer with 32 output units

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32))
