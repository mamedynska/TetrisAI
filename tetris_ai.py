import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import argparse
import os
import shutil
import tetris
import numpy as np
from random import random, randint, sample
from tetris import Tetris
from collections import deque


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
    model = DeepQNetwork()    # Notre réseau de neuronnes
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)     # Notre optimizer
    criterion = nn.MSELoss()        # Fonction de perte MeanSquar : va pénaliser le model pour de trop grande erreurs et encourage le model quand il en fait de petite

    state = env.reset()         # Remettre à zéro le tetris et retourner l'état 

    #Si on peut on utilise le GPU :
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    # Replay_memory =   la structure de données qui va nous servir de mémoire pour stocker nos ensembles (state, action, new_state, reward). 
    # C’est grâce à cette mémoire que l’on peut faire de l’experience replay. A chaque action, 
    # on va remplir cette mémoire au lieu d’entrainer, puis on va régulièrement piocher aléatoirement 
    # des samples dans cette mémoire, pour lancer l’entrainement sur un batch de données. 
    replay_memory = deque(maxlen=opt.replay_memory_size)

    # Génération = 0
    epoch = 0

    # Tant qu'on a pas fait toutes les générations
    while epoch < opt.num_epochs :
        # Récupére tous les états possibles à un instant t (avec la pièce disponible)
        next_steps = env.get_states()

        # Epsilon permet de soit faire une action random, soit une action choisie
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()

        # Booléen disant si c'est une action random ou choisie
        random_action = u <= epsilon

        # Listes des actions et des états
        next_actions, next_states = zip(*next_steps.items())

        # concatene tous les états en une ligne
        next_states = torch.stack(next_states)

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

        # On récupére depuis l'environnement la récompense après notre coup joué càd le score
        # Done est un booléen qui est à True si on a un gameover et False sinon
        reward, done = env.step(action, render=True)

        # GPU
        if torch.cuda.is_available():
            next_state = next_state.cuda()
        
        # Ajout dans la mémoire de notre action (avec le state, reward et done)
        replay_memory.append([state, reward, next_state, done])

        # Si gameover on enregistre le score final et on reset le tetris
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            
            # GPU
            if torch.cuda.is_available():
                state = state.cuda()

        # Sinon on continue
        else:
            state = next_state
            continue

        # Si on est arrivé au bon nombre de génération on test sinon on continue à boucler
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        
        # TODO : 
        epoch += 1
        # On pioche aléatoirement dans replay_memory une action/état
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        # On récupére chacun des attributs : 
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        # GPU
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()


        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))





# opt = get_args()
# epoch = 0

# epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
#                     opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
# u = random()
# random_action = u <= epsilon
# print(random_action)

if __name__ == "__main__":
    opt = get_args()
    train(opt)
