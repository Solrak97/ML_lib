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
    weights = []
    layers = []
    layer_dims = None
    learning_rate = None
    moemntum = None
    decay = None
    epochs = None
    seed = None
    trained = False
    weight_gradients = []


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
            self.weights.append(w_matrix)

        n = layers[-2] + 1
        m = layers[-1] 

        w_matrix = self.xaviers_init((n, m))
        self.weights.append(w_matrix)
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
        # Método step(self) que ejecuta la actualización de los pesos de la matriz, 

        # aplica la actualización de la tasa de aprendizaje en caso de decaimiento

        # aplica el momentum.
        
        # Además avanza el contador de época en 1.
        self.epochs += 1
        pass

    # Tridimensional space for fun purposes
    def delta_k(self, i, k, delta_i):
        delta = 0
        print(self.weights[i].shape)
        print(self.layers[i].neuronas)

        for j in range(self.layers[i + 1].neuronas):
            print(i,j,k)
            delta += self.weights[i][k][j] * delta_i[j] * self.layers[i].gradiente(self.layers[i].salida[j]) 
        return delta


    def backpropagation(self, x, y):
        p = self.predict(x)

        delta = 2 * (p - y) * self.layers[-1].gradiente(p)
        
        for i in range(len(self.layers) - 2, 0, -1):
            deltas = []
            for k in range(self.layers[i].neuronas):
                deltak = self.delta_k(i, k, delta)
                deltas.append(deltak)
            delta = np.array(deltas)




    def predict(self, x):
        if not self.trained:
            raise Exception('Error: No se pueden realizar predicciones sobre un modelo no entrenado')
        
        neto = x
        for i in range(len(self.weights)):
            neto = np.r_[neto, np.ones(1)]
            self.layers[i].set_netos(neto)
            salida = self.layers[i].activate()
            neto = np.matmul(salida, self.weights[i])

        self.layers[-1].set_netos(neto)
        salida = self.layers[-1].activate()

        return salida
        
