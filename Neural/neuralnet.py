from cmath import sqrt
import numpy as np
from layers import DenseLayer



def MSE(y_true, y_predict) -> float:
    n = y_true.shape[0]
    if n != y_predict.shape[0]:
        raise Exception("true and predict sizes don't match")
    
    else:
        return np.sum(((y_true - y_predict) ** 2)/n)


class DenseNN:
    wegths = []
    layers = []
    layer_dims = None
    learning_rate = None
    moemntum = None
    decay = None
    epochs = None
    seed = None
    trained = False

    def __init__(self, layers: list[int], activation: list[str], seed=0):
        self.layer_dims = layers
        self.crea_capas(layers, activation)
        self.seed = seed
        pass

    
    def crea_capas(self, layers, activation):

        entrada = DenseLayer(layers[0], tipo='entrada')
        self.layers.append(entrada)

        for i in range(1, len(layers) - 1):
            hidden = DenseLayer(layers[i], activacion=activation[i], tipo='hidden')
            self.layers.append(hidden)

        salida = DenseLayer(layers[-1], activation[-1], tipo='salida')
        self.layers.append(salida)
        pass


    def crea_pesos(self, layers, seed):
        np.random.seed(seed)
        for i in range(1, len(layers) - 1):
            n = layers[i - 1] + 1
            m = layers[i]

            w_matrix = self.xaviers_init((n, m))
            self.wegths.append(w_matrix)

        n = layers[-2] + 1
        m = layers[-1]

        w_matrix = self.xaviers_init((n, m))
        self.wegths.append(w_matrix)
        pass


    def xaviers_init(self, shape):
        fan_in, fan_out = shape
        weigth_matrix = np.random.normal(
            0, (2.0 / sqrt(fan_in + fan_out).real), size=(fan_in, fan_out))
        return weigth_matrix


    def train(self, lr=0.05, momentum=0, decay=0):
        self.trained = True
        self.epochs = 0
        self.crea_pesos(self.layer_dims, self.seed)
        pass

    def step(self):
        pass

    def backpropagation(self, x, y):
        p = self.predict(x)
        e = MSE(y, p)
        print(e)
        pass


    def predict(self, x):
        if not self.trained:
            raise Exception('Error: No se pueden realizar predicciones sobre un modelo no entrenado')
        # Inicializa el valor de entrada
        neto = x

        # Recorre n - 1 capas, la primera tiene tag entrada
        # Esto causa que la activación retorne el valor neto
        for i in range(len(self.wegths)):
            neto = np.c_[neto, np.ones(x.shape[0])]
            self.layers[i].set_netos(neto)
            salida = self.layers[i].activate()
            neto = np.matmul(salida, self.wegths[i])

        # Se obtiene el reulstado aplicando la ultima capa
        self.layers[-1].set_netos(neto)
        salida = self.layers[-1].activate()

        return salida
        
