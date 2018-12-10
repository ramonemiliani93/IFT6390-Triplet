import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm

import utils
from metrics import dict_metrics
from model.net import LinearRegression, MLP, CNN
import model.data_loader as data_loader
from evaluate import evaluate_crossentropy, evaluate_triplets
from losses import TripletLoss

from tensorboardX import SummaryWriter

plt.style.use('seaborn-white')
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

parser = argparse.ArgumentParser()
parser.add_argument('model', default=None, choices=['linear', 'mlp', 'cnn'], help="Model to train")
parser.add_argument('dataset', default=None, choices=['fashion', 'cifar'], help="Model to train")
parser.add_argument('checkpoint', default=None, help='Model weights')
parser.add_argument('--data_dir', default='data', help="Directory that will contain dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def plot_embeddings_3D(embeddings, targets):

    pca = decomposition.PCA(n_components=3)
    pca.fit(embeddings)
    x = pca.transform(embeddings)

    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    # for i in np.unique(targets):
    ax.scatter(x[:, 0],
               x[:, 1],
               x[:, 2], c=targets, cmap=plt.cm.nipy_spectral, edgecolor='k')

    plt.show()


def plot_embeddings_2D(embeddings, targets, model, dataset, loss):

    pca = decomposition.PCA(n_components=2)
    pca.fit(embeddings)
    x = pca.transform(embeddings)

    plt.figure(1, figsize=(8, 6))
    plt.clf()

    # for i in np.unique(targets):
    plt.scatter(x[:, 0],
               x[:, 1], c=targets, cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.tight_layout()
    plt.savefig('{}_{}_{}'.format(model, dataset, loss))
    plt.show()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset.indices), 50))
        labels = np.zeros(len(dataloader.dataset.indices))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings[k:k+len(images[0])] = model.extract_features(images[0]).data.cpu().numpy()
            labels[k:k+len(images[0])] = target.numpy()
            k += len(images[0])
    return embeddings, labels


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Hyperparameters(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'lg.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl, val_dl = data_loader.fetch_train_dataloaders(args.dataset, args.data_dir, params)
    test_dl = data_loader.fetch_test_dataloader(args.dataset, args.data_dir, params)

    logging.info("- done.")

    # Define the model and optimizer
    choices = {
        'linear': LinearRegression().cuda() if params.cuda else LinearRegression(),
        'mlp': MLP().cuda() if params.cuda else MLP(),
        'cnn': CNN().cuda() if params.cuda else CNN()
    }
    model = choices[args.model]
    utils.load_checkpoint(args.checkpoint, model)

    train_embeddings, train_labels = extract_embeddings(train_dl, model)
    val_embeddings, val_labels = extract_embeddings(val_dl, model)
    test_embeddings, test_labels = extract_embeddings(test_dl, model)

    plot_embeddings_2D(test_embeddings, test_labels, args.model, args.dataset, 'triplet')