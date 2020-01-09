import numpy as np


class Perceptron():
    '''Uma rede neural do tipo perceptron'''


    def __init__ (self, layers: tuple, learning_rate: float = 0.4):

        pattern = [(a, b) for a, b in zip(layers[:-1], layers[1:])]
        self.weigths = [np.random.rand(*a) for a in pattern]
        self.layers_shape = layers
        self.weigths_shape = pattern

        self.learning_rate = learning_rate



    @staticmethod
    def _sigmoid (x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)


    def predict(self, feed: tuple):

        layers_result = [np.array(feed)]

        for layer in self.weigths:

            result = np.dot(np.transpose(layer), layers_result[-1])
            result = self._sigmoid(result)
            
            layers_result.append(result)

        return np.array(layers_result)


    def back_propagation(self, feed:tuple, expected_output:tuple):
        
        layers_result = self.predict(feed)

        error = np.array(expected_output - layers_result[-1])
        delta = np.array(self.learning_rate * error * self._sigmoid_derivative(layers_result[-1]))

        for n in range(len(layers_result)-2, -1, -1):

            self.weigths[n] += np.dot(np.transpose([layers_result[n]]), [delta])

            error = np.dot(delta, self.weigths[n].T)
            delta = self.learning_rate * error * self._sigmoid_derivative(layers_result[n])

    def train(self, inputs: tuple, correct_outputs: tuple):

        for n in range(len(inputs)):
            self.back_propagation(inputs[n], correct_outputs[n])

