


# %tensorflow_version 2.x
from os import path
import h5py as h5
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras import Sequential
from tensorflow.keras import layers
from keras.applications.vgg19 import vgg19
from keras.applications import VGG16
from keras.models import load_model
from keras.engine import InputLayer
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.applications.vgg19 import VGG19
from keras.models import Model, Sequential


class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample

        """
        self.input_dimension = []
        self.model = Sequential()
        self.cnt = 0
        self.shp = ()
        self.wts = []
        self.biases = []
        self.layer_no = 0
        self.layer_shape = []
        self.nowts = []
        self.layerList = []
        self.model_load = "q"


    def add_input_layer(self, shape=(2,),name="" ):
        """
         This function adds a dense layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """

        shape_new=((shape))
        old_layer = "dsx"
        if self.cnt <= 0:
            if type(shape) == int or (type(shape)==tuple and len(shape)==1):
                if type(shape)==tuple :
                    shape_new, = shape
                self.model.add(InputLayer(input_shape=(shape_new,), name=name))
                self.layer_shape.insert(self.layer_no,shape_new)

            else:
                x, y, a = shape
                self.model.add(InputLayer(input_shape=shape,name=name))
                self.layer_shape.insert(self.layer_no, shape)
            self.cnt = self.cnt + 1

        else:
            if len(self.shp) == 1:
               m = np.array(shape)
               m = np.expand_dims(m, axis=1)
               m = np.array(m)

            else:
                x, y, a = tuple((self.shp))
            self.model.layers.pop(0)
            self.model.add(InputLayer(input_shape=shape, name=name))
            self.layer_shape.insert(self.layer_no ,shape)

        self.wts.insert(self.layer_no, 0)
        self.biases.insert(self.layer_no, 0)
        self.nowts.insert(self.layer_no,0)
        if name == None or name =='':
            name ="nowtslayer"
        self.layerList.insert(self.layer_no, name)
        self.layer_no = self.layer_no + 1

    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        if self.model.name != "vgg19" and self.model.name != "vgg16":
            self.model.add(Dense(num_nodes, activation=activation,name=name))
            self.layer_shape.insert(self.layer_no, num_nodes)
            self.cnt = self.cnt+1
            j = self.layer_no
            a = self.layer_shape[self.layer_no - 1]
            for i in range(self.layer_no):
                if self.nowts[i] != 0:
                    a = self.layer_shape[i]

            self.nowts.insert(self.layer_no, 1)
            self.wts.insert(self.layer_no,tf.Variable(initial_value=tf.random_uniform_initializer()(shape=(a, num_nodes),dtype='float32'),trainable=trainable))
            self.biases.insert(self.layer_no, tf.Variable(initial_value=tf.random_uniform_initializer()(shape=(num_nodes,), dtype='float32'),
                                                                trainable=trainable))
            self.layer_no = self.layer_no + 1
        else:
            self.model = Model(self.model.input, Dense(num_nodes, activation=activation)(self.model.output))

        if name == None or name =='':
            name ="nowtslayer"
        self.layerList.insert(self.layer_no, name)


    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        self.model.add(Conv2D(num_of_filters, kernel_size, strides=(strides, strides), activation="relu",padding =padding,name=name))
        self.layer_shape.insert(self.layer_no, num_of_filters)
        self.nowts.insert(self.layer_no, 2)
        if name == None or name =='':
            name = "nowtslayer"
        self.layerList.insert(self.layer_no, name)
        if type(self.layer_shape[self.layer_no - 1]) == int and type(kernel_size) != int:
            self.wts.insert(self.layer_no, tf.Variable(
                initial_value=tf.random_uniform_initializer()(shape=(kernel_size[0], kernel_size[1], self.layer_shape[self.layer_no - 1], num_of_filters),
                                     dtype='float32'), trainable=trainable))
            self.biases.insert(self.layer_no, tf.Variable(initial_value=tf.random_uniform_initializer()(shape=(num_of_filters,), dtype='float32'),
                                                              trainable=trainable))


        elif type(self.layer_shape[self.layer_no - 1]) == int and type(kernel_size) == int:
            self.wts.insert(self.layer_no, tf.Variable(
                initial_value=tf.random_uniform_initializer()(shape=(kernel_size, kernel_size, self.layer_shape[self.layer_no - 1], num_of_filters),
                                     dtype='float32'), trainable=trainable))
            self.biases.insert(self.layer_no, tf.Variable(initial_value=tf.random_uniform_initializer()(shape=(num_of_filters,), dtype='float32'),
                                                              trainable=trainable))
        else:
            self.wts.insert(self.layer_no, tf.Variable(
                initial_value=tf.random_uniform_initializer()(shape=(kernel_size[0],kernel_size[1],self.layer_shape[self.layer_no - 1][-1],num_of_filters),
                                     dtype='float32'), trainable=trainable))
            self.biases.insert(self.layer_no,
                               tf.Variable(initial_value=tf.random_uniform_initializer()(shape=(num_of_filters,), dtype='float32'),
                                           trainable=trainable))
        self.layer_no = self.layer_no + 1
        self.cnt =self.cnt + 1
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        if name == None or name == '':
            name = "nowtslayer"
        self.model.add(MaxPooling2D(pool_size=pool_size, strides=(strides, strides), padding=padding,name=name))
        self.nowts.insert(self.layer_no, 0)
        self.layerList.insert(self.layer_no, name)
        self.layer_shape.insert(self.layer_no, pool_size)
        self.wts.insert(self.layer_no, 0)
        self.biases.insert(self.layer_no, 0)
        self.cnt = self.cnt + 1
        self.layer_no = self.layer_no + 1

    def group(self,obj):
        if isinstance(obj, h5.Group):
            return True
        return False


    def append_flatten_layer(self,name=""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        if name == None or name =='':
            name = "nowtslayer"
        self.model.add(Flatten(name=name))
        self.layerList.insert(self.layer_no, name)
        self.wts.insert(self.layer_no, 0)
        self.biases.insert(self.layer_no, 0)
        self.nowts.insert(self.layer_no, 0)
        self.cnt = self.cnt + 1
        self.layer_no = self.layer_no + 1




    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This function sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        return None



    def getDatasetFromGroup(self,datasets, obj):
        if self.group(obj):
            for key in obj:
                x = obj[key]
                self.getDatasetFromGroup(datasets, x)
        else:
            datasets.append(obj)


    def getWeighthsForLayer(self,layerName, fileName):
        weigths = []
        with h5.File(fileName, mode='r') as fk:
            for key in fk:
                if layerName in key:
                    datasets = []
                    self.getDatasetFromGroup(datasets, fk[key])
                    for dataset in datasets:
                        weigths.append(np.array(dataset))
        for w in weigths:
            print("w shape",w.shape)
        return w

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """

        if self.model.name != "vgg19" and  self.model.name != "vgg16":
            if layer_name != "nowtslayer":
                for i, j in enumerate(self.layerList):
                    if j == layer_name:
                        layer_number= i

            if layer_number == None:
                layer_number = self.layer_no-1
            if self.nowts[layer_number] == 0:
                return None
            else:
                return self.wts[layer_number]
        else:
            if layer_name == "":
                return self.model.layers[layer_number].get_weights()[0]
            else:
                return self.getWeighthsForLayer(layer_name,self.model_load+".h5")

    def get_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """
        flag = 0
        if layer_name != "":
            flag = 1
        if layer_name != "nowtslayer":
            for i, j in enumerate(self.layerList):
                if j == layer_name:
                    layer_number= i

        if layer_number == None:
            layer_number = self.layer_no - 1
        if self.nowts[layer_number] == 0:
            return None
        else:
            if flag != 1 :
                return self.biases[layer_number]
            else:
                sz = self.biases[layer_number].shape
                a = []
                y = 1
                x, = sz
                for y in range(x - 1):
                    a.insert(y, self.biases[layer_number][y])
                return a

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
        if layer_number == None:
            layer_number = self.layer_no - 1

        if layer_name != "nowtslayer":
            for i, j in enumerate(self.layerList):
                if j == layer_name:
                    layer_number= i
        self.wts[layer_number] = weights

    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        if layer_number == None:
            layer_number = self.layer_no - 1

        if layer_name != "nowtslayer":
            for i, j in enumerate(self.layerList):
                if j == layer_name:
                    layer_number= i
        self.biases[layer_number] = biases

    def pop_last_layer(self):
        """
        This function removes a layer from the model and connects the previous and next layer
        (if they exist).
        :return: poped layer
        """
        self.model=keras.Model(inputs=self.input_layer_tensor,outputs=self.model.layers[-2].output)
        return self.model.layers.pop()

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        self.model =Sequential()
        self.model_load  = model_name
        if model_name == "":
            self.model_load =  model_file_name

        if model_file_name == "":
            if model_name == "VGG16":
                self.model = VGG16()
            else:
                self.model = VGG19()
        else:

             self.model = load_model(model_file_name)
             if model_name == "VGG16":
                 self.model.name ="vgg16"
             else:
                 self.model.name ="vgg19"
        self.model.save_weights(self.model_load+".h5")
        return self.model



    def save_model(self,model_file_name=""):
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """

        self.model.save_weights(model_file_name)
        self.model.save(model_file_name)
        return self.model


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """
        self.model.compile(loss=loss)
        return None


    def set_metric(self,metric):
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        self.model.compile(metrics=[metric])
        return None


    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
        self.model.compile(optimizer=optimizer,lr = learning_rate,momentum=momentum)
        return  None


    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        predictvalue = self.model.predict(X)
        return predictvalue


    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        loss, accuracy = self.model.evaluate(X,y)
        return loss, accuracy

    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param X_validation: Array of input validation data
         :param y: Array of desired (target) validation outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
        self.model.fit(X_train, y_train, epochs=num_epochs,batch_size=batch_size)


if __name__ == "__main__":

    my_cnn=CNN()
    print(my_cnn)
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    # my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='linear',name="conv1")
    # print(my_cnn.model.summary())
    weights=my_cnn.get_weights_without_biases(layer_number=0)
    biases=my_cnn.get_biases(layer_number=0)
    print("w0",None if weights is None else weights.shape,type(weights))
    print("b0",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=1)
    biases=my_cnn.get_biases(layer_number=1)
    print("w1",None if weights is None else weights.shape,type(weights))
    print("b1",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=2)
    biases=my_cnn.get_biases(layer_number=2)
    print("w2",None if weights is None else weights.shape,type(weights))
    print("b2",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=3)
    biases=my_cnn.get_biases(layer_number=3)
    print("w3",None if weights is None else weights.shape,type(weights))
    print("b3",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=4)
    biases=my_cnn.get_biases(layer_number=4)
    print("w4",None if weights is None else weights.shape,type(weights))
    print("b4",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_number=5)
    biases = my_cnn.get_biases(layer_number=5)
    print("w5", None if weights is None else weights.shape, type(weights))
    print("b5", None if biases is None else biases.shape, type(biases))

    weights=my_cnn.get_weights_without_biases(layer_name="input")
    biases=my_cnn.get_biases(layer_number=0)
    print("input weights: ",None if weights is None else weights.shape,type(weights))
    print("input biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv1")
    biases=my_cnn.get_biases(layer_number=1)
    print("conv1 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="pool1")
    biases=my_cnn.get_biases(layer_number=2)
    print("pool1 weights: ",None if weights is None else weights.shape,type(weights))
    print("pool1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv2")
    biases=my_cnn.get_biases(layer_number=3)
    print("conv2 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv2 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="flat1")
    biases=my_cnn.get_biases(layer_number=4)
    print("flat1 weights: ",None if weights is None else weights.shape,type(weights))
    print("flat1 biases: ",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense1")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense1 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense1 biases: ", None if biases is None else biases.shape, type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense2")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense2 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense2 biases: ", None if biases is None else biases.shape, type(biases))
