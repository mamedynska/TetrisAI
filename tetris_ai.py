import torch
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        input = 10
        output = output2 = output3 = 10
        finalDecision = 6
        self.fc1 = nn.Linear(input, output)         # Nb input = le nombre de valeur à prendre en compte pour la décision 
                                                    # Nb output le nombre de choix possible pour une décision
        self.fc2 = nn.Linear(output, output2)       # 1Deuxième couche de neuronnes
        self.fc3 = nn.Linear(output2, output3)      # Troisième : Possibilité que output=output2=output3
        self.fc4 = nn.Linear(output3, finalDecision)


# Pour chaque données on l'a fait passer à travers notre réseau de neuronnes
    def forward(self, x):
        x = F.relu(self.fc1(x))         # F.relu transforme notre x (qui peut être = 15605 par ex) dans un intervalle [0,1]
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)                 # Pour notre dernière couche qui sera une matrice de taille [1, nb de coup jouable] 

        return x

net = Net()
print(net)

x = torch.rand((1,10))
print(x)
x = x.view(1,10)
output=net(x)
print(output)