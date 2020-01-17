import numpy as np
import os
import json


np.seterr(over='ignore')

class Perceptron():
    '''Uma rede neural do tipo perceptron'''
    def __init__ (self, layers: tuple, learning_rate: float = 0.4):

        pattern = [(a, b) for a, b in zip(layers[:-1], layers[1:])]
        self.weigths = [np.random.randn(*x).astype(np.float32) for x in pattern]
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

        for layer in range(len(self.weigths_shape)):

            result = np.dot(np.transpose(self.weigths[layer]), layers_result[-1])
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

    def save_brain (self, fp: str, overwrite: bool = False):
        if not overwrite and os.path.exists(fp):
            raise FileExistsError("The Brain already exists")

        with open(fp, "w") as file:
            attributes = self.__dict__
            attributes = dict(map(lambda x: [x[0], x[1].tolist()] if type(x[1]) == "ndarray" else x, attributes.items()))

            for weigth in range(len(attributes['weigths'])):
                attributes['weigths'][weigth] = attributes['weigths'][weigth].tolist()
            
            json.dump(attributes, file)

    @classmethod
    def load_brain (cls, fp: str):
        if not os.path.exists(fp):
            raise FileNotFoundError("Brain not found")

        with open(fp, "r") as file:
            attributes = json.load(file)

        self = Perceptron.__new__(cls)
        for name, value in attributes.items():
            setattr(self, name, value)


        return self
             
        

