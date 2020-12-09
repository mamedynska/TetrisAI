import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import argparse
import os
import shutil
import tetris
import numpy as np
from random import random, randint, sample


class DeepQNetwork(nn.Module):
    
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))     # Sequential : Linear and ReLu
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))    # Sequential : Linear and ReLu
        self.conv3 = nn.Sequential(nn.Linear(64, 1))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)             # Loi uniforme
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x) # Au lieu de faire le RELU ici on le fait avant dans __init__
        x = self.conv2(x)
        x = self.conv3(x)

        return x


def get_args():
    # Paramètre de la grid Tetris
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    # parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    # parser.add_argument("--height", type=int, default=15, help="The common height for all images")
    # parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    # parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    # parser.add_argument("--lr", type=float, default=1e-3)                   # Learning rate  : permet de dire à l'optimizer de ne pas sur apprendre sur chaque données
    #                                                                         # On ne va pas que lorsqu'il se trompe dans sa prédiction, il se corrige pour avoir une probabilité de 0 
    #                                                                         # partout et de 1 sur le bon chiffre
    #                                                                         # Sinon on va sur-apprendre et ce n'est pas bon.
    #                                                                         # On va donc optimiser la perte "Loss" sans la rendre nulle pour éviter le sur-apprentissage
    # parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--initial_epsilon", type=float, default=1)
    # parser.add_argument("--final_epsilon", type=float, default=0.001)
    # parser.add_argument("--num_decay_epochs", type=float, default=2000)
    # parser.add_argument("--num_epochs", type=int, default=3000)             # Nombre de génération
    # parser.add_argument("--save_interval", type=int, default=1000)          # Intervalle de sauvegarde du réseau
    # parser.add_argument("--replay_memory_size", type=int, default=30000,    
    #                     help="Number of epoches between testing phases")
    # parser.add_argument("--log_path", type=str, default="tensorboard")      # Chemin des logs
    # parser.add_argument("--saved_path", type=str, default="trained_models") # Chemin de sauvegarde
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=15, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=0.001)                   # Learning rate  : permet de dire à l'optimizer de ne pas sur apprendre sur chaque données
                                                                            # On ne va pas que lorsqu'il se trompe dans sa prédiction, il se corrige pour avoir une probabilité de 0 
                                                                            # partout et de 1 sur le bon chiffre
                                                                            # Sinon on va sur-apprendre et ce n'est pas bon.
                                                                            # On va donc optimiser la perte "Loss" sans la rendre nulle pour éviter le sur-apprentissage
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=0.001)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)             # Nombre de génération
    parser.add_argument("--save_interval", type=int, default=1000)          # Intervalle de sauvegarde du réseau
    parser.add_argument("--replay_memory_size", type=int, default=30000, help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")      # Chemin des logs
    parser.add_argument("--saved_path", type=str, default="trained_models") # Chemin de sauvegarde
    args = parser.parse_args()
    return args

# Fonction d'apprentissage
def train(opt):
    # Cuda permet d'éxécuter sur le GPU, si on peut le faire on l'utilise
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    # Sinon ça sera sur le CPU
    else:
        torch.manual_seed(123)
    # Récupération d'un log
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris()          # Notre jeu tetris, ici c'est l'environnement de l'IA d'où "env"
    net = DeepQNetwork()    # Notre réseau de neuronnes
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)     # Notre optimizer
    criterion = nn.MSELoss()        # Fonction de perte MeanSquar : va pénaliser le model pour de trop grande erreurs et encourage le model quand il en fait de petite

    state = env.reset()         # Remettre à zéro le tetris et retourner l'état 

    #Si on peut on utilise le GPU :
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    # Génération = 0
    epoch = 0
    # Tant qu'on a pas fait toutes les générations
    while epoch < opt.num_epochs :
        # next_steps = env.get_next_states()

        # Epsilon permet de soit faire une action random, soit une action choisie
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()

        # Booléen disant si c'est une action random ou choisie
        random_action = u <= epsilon

        # Listes des actions et des états
        next_actions, next_states = zip(*next_steps.items())

        # concatene tous les états en une ligne
        next_states = torch.stack(next_states)*

        # GPU
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # model.eval() notifiera à toutes nos couches que nous sommes en mode évaluation, de cette façon, 
        # les couches batchnorm ou dropout fonctionneront en mode évaluation au lieu du mode apprentissage.
        model.eval()
        # torch.no_grad() agit sur le moteur autograde et le désactive. 
        # Cela réduira l'utilisation de la mémoire et accélérera les calculs 
        # mais on ne pourra pas faire de backprop (ce qu'on ne veut pas dans un script d'évaluation).
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        
        # model.train() indique à votre modèle que vous êtes en train de lui apprendre. Ainsi, les couches telles que dropout, batchnorm reprennent leur fonctionnement en mode apprentissage
        model.train()

        # Soit action random, soit action prédite
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        # Etat et action correspondant au choix
        next_state = next_states[index, :]
        action = next_actions[index]



opt = get_args()
epoch = 0

epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
u = random()
random_action = u <= epsilon
print(random_action)
