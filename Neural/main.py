import pandas as pd
from layers import DenseLayer

from neuralnet import DenseNN



base_dataset = pd.read_csv("Data/titanic.csv")

titanic_data = base_dataset.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
titanic_data.dropna(axis=0, how="any", inplace=True)

cnames = ["Pclass", "Sex", "Embarked"]
for cname in cnames:
    dummies = pd.get_dummies(titanic_data[cname], prefix=cname)
    titanic_data = titanic_data.drop(cname, axis=1)
    titanic_data = titanic_data.join(dummies)




y = titanic_data['Survived']
X = titanic_data.drop(columns='Survived')


layers = [12,7,5,2,1]
activation = ['l','l','l','s']

nn = DenseNN(layers, activation, seed=0)
nn.train()
pred = nn.predict(X)

print(pred.shape)
