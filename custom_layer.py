from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class My_layer(Layer):
#class My_Layer:

    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super(My_layer, self).__init__(**kwargs)

    def call(self, x):
        return K.reshape(K.sum(x, axis=(1,2)),(-1,1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'output_dim': self.output_dim})
        return config 
