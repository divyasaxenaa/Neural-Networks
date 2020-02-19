

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix as cm


def display_images(images):
	# This function displays images on a grid.
	# Farhad Kamangar Sept. 2019
	number_of_images=images.shape[0]
	number_of_rows_for_subplot=int(np.sqrt(number_of_images))
	number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
	for k in range(number_of_images):
		plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
		plt.imshow(images[k], cmap=plt.get_cmap('gray'))
		# plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
	plt.show()

def display_numpy_array_as_table(input_array):
	# This function displays a 1d or 2d numpy array (matrix).
	# Farhad Kamangar Sept. 2019
	if input_array.ndim==1:
		num_of_columns,=input_array.shape
		temp_matrix=input_array.reshape((1, num_of_columns))
	elif input_array.ndim>2:
		print("Input matrix dimension is greater than 2. Can not display as table")
		return
	else:
		temp_matrix=input_array
	number_of_rows,num_of_columns = temp_matrix.shape
	plt.figure()
	tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
	for cell in tb.properties()['child_artists']:
	    cell.set_height(1/number_of_rows)
	    cell.set_width(1/num_of_columns)

	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()


class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions+1
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def linear(self, x):
        return x

    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions)

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions))

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
        """

        X = np.insert(X, 0, 1, axis=0)
        W = np.dot(self.weights, X)
        #print("w")
        #print(W)
        Num_array = np.array(W)
        NUm_array_hard = []
        if self.transfer_function == "Hard_limit":
            for list in W:
                for number in list:
                    if number > 0:
                        NUm_array_hard.append(1)
                    else:
                        NUm_array_hard.append(0)
            #print("NUm_array_hard")
            NUm_array_hard = np.reshape(NUm_array_hard, np.shape(Num_array))
            #print(NUm_array_hard)
            #print("***predict end****")
            return NUm_array_hard
        elif self.transfer_function == "Linear":
            NUm_array_linear = self.linear(np.dot(self.weights, X))
            #print("NUm_array_linear", NUm_array_linear)
            return NUm_array_linear
        else:
            NUm_array_sigmoid = self.sigmoid(np.dot(self.weights, X))
            #print("NUm_array_sigmoid", NUm_array_sigmoid)
            return NUm_array_sigmoid
    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """



    def calculate_percent_error(self,X, y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        output_pred = self.predict(X)
        #print("**************%%%%%***********")
        counter = 0
        percent_err_percent = 0
        # print(",X.shape[0]" ,output_pred.shape[0])
        y_predict = self.predict(X)
        classpredicted = np.argmax(y_predict, axis=0)
        #print("classpredicted", classpredicted)
        classactual = y
        """np.argmax(y, axis=0)"""
        #print("classactual", classactual)
        number_of_samples = y.shape[0]
        #print("number_of_samples", number_of_samples)
        counter = 0
        for i in range(number_of_samples):
           if classpredicted[i] != classactual[i]:
                counter = counter + 1
        percent_err_percent = counter / number_of_samples
        return percent_err_percent

    def calculate_confusion_matrix(self,X,y):
        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """

        confusion_matrix = np.zeros((self.number_of_classes, self.number_of_classes))
        for j in range(self.number_of_classes):
            for k in range(self.number_of_classes):
                confusion_matrix[j][k] = 0
        print("confusion_matrix", np.shape(confusion_matrix))
        print("confusion_matrix", confusion_matrix)
        y_predict = self.predict(X)
        classpredicted = np.argmax(y_predict, axis=0)
        #print("classpredicted", classpredicted)
        classactual = y
        """np.argmax(y, axis=0)"""
        #print("classactual", classactual)
        number_of_samples = y.shape[0]
        #print("number_of_samples", number_of_samples)
        for i in range(number_of_samples):
            rowno = classpredicted[i]
            colno = classactual[i]
            #print("classpred", classpredicted[i])
            #print("classactual", classactual[i])
            # print("confusion_matrix[rowno][colno]", confusion_matrix[rowno][colno])
            confusion_matrix[colno][rowno] = confusion_matrix[colno][rowno] + 1
            # confusion_matrix[rowno][colno]
        #print("confusion_matrix[ ", confusion_matrix)
        return confusion_matrix

if __name__ == "__main__":

    # Read mnist data
    number_of_classes = 10
    number_of_training_samples_to_use = 700
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    number_of_images_to_view=16
    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    display_images(test_x)
    input_dimensions=X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=5)
    # model.initialize_all_weights_to_zeros()
    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error)
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print("X_test_vectorized",np.shape(X_test_vectorized))
    print("y_test", np.shape(y_test))
    print(np.array2string(confusion_matrix, separator=","))
