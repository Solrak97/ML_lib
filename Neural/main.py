import pandas as pd
import numpy as np
from layers import DenseLayer

from neuralnet import DenseNN


def train(net, epochs, x, y):
    net.train()

    for epoch in range(1, epochs + 1):
        net.backpropagation(x, y)
        net.step()

        print(f'Epoch #{epoch}')


base_dataset = pd.read_csv("Data/titanic.csv")

titanic_data = base_dataset.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
titanic_data.dropna(axis=0, how="any", inplace=True)

cnames = ["Pclass", "Sex", "Embarked"]
for cname in cnames:
    dummies = pd.get_dummies(titanic_data[cname], prefix=cname)
    titanic_data = titanic_data.drop(cname, axis=1)
    titanic_data = titanic_data.join(dummies)




y = titanic_data['Survived'].to_numpy()
X = titanic_data.drop(columns='Survived')
y = y.reshape(-1 , 1)

layers = [12,7,5,3]
activation = ['l','l','l','s']

nn = DenseNN(layers, activation, seed=0)

train(nn, 1000, X, y)

pred = nn.predict(X)
print(pred.shape)
