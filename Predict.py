from IA import Perceptron
from PIL import Image, ImageOps
import numpy as np

def Predict(neural_network: Perceptron, file_name: str):
    image = Image.open(file_name)
    image_array = np.array(image)
    image_array = image_array.reshape(image_array.shape[0]*image_array.shape[1])

    result = neural_network.predict(image_array)[-1].tolist()

    return result










if __name__ == "__main__":

    file_name = input("Diga o nome da imagem a ser lida: ")
    neural_network = Perceptron.load_brain("./Brain.dat")
    print(Predict(neural_network, file_name))

    image = Image.open(file_name)
    image.show()

    

