

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions+1
        self.number_of_classes = number_of_classes
        self.seed = seed
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = []
        #self.weights = np.random.normal(0, self.seed, (self.number_of_classes, self.input_dimensions))
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions)
        #print("self.weights ini")
        #print(self.weights)
        #raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = []
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions))
        #raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        #print("predict start")
        X = np.insert(X, 0, 1, axis=0)
        #print("X after")
        #print(X)
        #print("weights")
        #print(self.weights)
        #print("shape x", np.shape(X))
        #print("shape wt", np.shape(self.weights))
        W = np.dot(self.weights, X)
        #print("w")
        #print(W)
        Num_array = np.array(W)
        NUm_array_hard = []
        for list in W:
            for number in list:
                if number > 0:
                    NUm_array_hard.append(1)
                else:
                    NUm_array_hard.append(0)
        #print("NUm_array_hard")
        NUm_array_hard = np.reshape(NUm_array_hard,np.shape(Num_array))
        #print(NUm_array_hard)
        #print("***predict end****")
        return NUm_array_hard
        #raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        #raise Warning("You must implement print_weights")

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        #print("train start")
        for it in range(num_epochs):
                X = np.insert(X, 0, 1, axis=0)
                for x in range(X.shape[1]):
                    X_dime = np.expand_dims(X[:, x], axis=1)
                    #print("X_dime", X_dime)
                    total = np.dot(self.weights, X_dime)
                    output = np.where(total >= 0, 1, 0)
                    #print("output", output)
                    Y_dime = np.expand_dims(Y[:, x], axis=1)
                    error = Y_dime - output
                    #print("error", error)
                    error_final = (np.dot((error), np.expand_dims(X[:, x], axis=1).T))
                    #print("error_final", error_final)
                    self.weights = ((self.weights) + (error_final * alpha))
            #raise Warning("You must implement train")

        #print("self.weights train", self.weights)
        #print("train end")
    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        #print("error start")
        output_pred = self.predict(X)
        #print("output_pred", output_pred)
        counter = 0
        for j in range(X.shape[1]):
             y_slice = Y[:, j]
             Y_fix = np.expand_dims(y_slice, axis=1)
             wt_slice = output_pred[:, j]
             wt_fix = np.expand_dims(wt_slice, axis=1)
             if np.array_equal(wt_fix, Y_fix):
                 counter = counter + 1
        #print("counter", counter)
        percent_err_percent = (X.shape[1]-counter)/X.shape[1]
        #print("percent_err_percent",percent_err_percent)
        return percent_err_percent
    #print("error end")

        #raise Warning("You must implement calculate_percent_error")

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)
