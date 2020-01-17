from IA import Perceptron
from keras.datasets import mnist
import numpy as np
import os


def Train (neural_network: Perceptron, inputs: tuple, outputs:tuple, save_fp: str, times: int = 1):
    
    size = 50
    for i in range(times):
        print("\x1b[K[{}{}] {}%".format( "#"*int(size*i/times), "-"*int(size-size*i/times), round(100*i/times, 2)), end="\r")
        neural_network.train(inputs, outputs)
        neural_network.save_brain(save_fp, overwrite=True)

def Test (neural_network: Perceptron, feed: np.ndarray, label: int):
    result = neural_network.predict(feed)[-1]

    print(f'Input: {label}, Output: {result}')



(inputs, outputs) = mnist.load_data()[0]


brain_file_path = "./Brain.dat"

outputs_one_hot = np.array([np.zeros(10) for x in outputs])
for i in range(len(outputs)):
    outputs_one_hot[i, outputs[i]] = 1


size = inputs.shape[1]*inputs.shape[2]


if os.path.exists(brain_file_path):
    neural_network = Perceptron.load_brain(brain_file_path)
else:
    neural_network = Perceptron((size, (size+10)//2, (size+10)//2, 10), learning_rate=1)

Train(neural_network, inputs.reshape(inputs.shape[0], size), outputs, brain_file_path, 1)






