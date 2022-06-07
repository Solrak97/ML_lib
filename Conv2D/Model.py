from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Softmax
from torch import flatten, sigmoid
from torch.nn.functional import relu
from utils import plot_tensor


class CnnModel(Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        # Primer capa convolucional:
        self.conv1 = Conv2d(in_channels=3, out_channels=40, kernel_size=(3,3))
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Segunda capa convolucional
        self.conv2 = Conv2d(in_channels=40, out_channels=80, kernel_size=(3,3))
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        
        # Tercera capa convolucional
        self.conv3 = Conv2d(in_channels=80, out_channels=160, kernel_size=(3,3))
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        # Corregir el tamaño de las dimensiones en el futuro
        # Capa de aprendizaje
        self.fc1 = Linear(in_features=40, out_features=80)
        self.fc2 = Linear(in_features=60, out_features=30)

        # Capa de clasificación
        self.fc3 = Linear(in_features=80, out_features=120)
        self.softmax = Softmax(dim=0)
        
    
    
    def forward(self, x):
        # Capa 1
        x = relu(self.conv1(x))
        x = self.maxpool1(x)
        
        # Capa 2
        x = relu(self.conv2(x))
        x = self.maxpool2(x)

        # Capa 3
        x = relu(self.conv3(x))
        x = self.maxpool3(x)

        print(x.shape)
        
        # Flattening
        x = x.view(-1, 120)
        
        # FC1
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        
        # Clasificación
        x = self.fc3(x)
        output = self.softmax(x)

        return output
        