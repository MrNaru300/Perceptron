import numpy as np
import IA
from keras.datasets import mnist



(feed, lambels), (feed_test, lambels_test) = mnist.load_data()

print("Treinando {} images com {}/{} de tamanho".format(feed.shape[0], feed.shape[1], feed.shape[2]))

size = feed.shape[1]*feed.shape[2]

neural_network = IA.Perceptron((size, int(size/2), int(size/3), int(size/6), 1))
feed = feed.reshape(feed.shape[0], size)


neural_network.train(feed, lambels)

print("Terminado")