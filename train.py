"""Train the model"""

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

from tensorboardX import SummaryWriter

# Constants
SAVE_SUMMARY_STEPS = 100


parser = argparse.ArgumentParser()
parser.add_argument('model', default=None, choices=['linear', 'mlp', 'cnn'], help="Model to train")
parser.add_argument('dataset', default=None, choices=['fashion', 'cifar'], help="Model to train")
parser.add_argument('loss', default=None, choices=['crossentropy', 'hinge', 'mse', 'triplet'], help="Model to train")
parser.add_argument('--data_dir', default='data', help="Directory that will contain dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train_other(model, optimizer, loss_fn, metrics, train_batch, labels_batch, summ, i):

    # move to GPU if available
    if params.cuda:
        train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

    # convert to torch Variables
    train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

    # compute model output and loss
    output_batch = model(train_batch)
    loss = loss_fn(output_batch, labels_batch)

    # clear previous gradients, compute gradients of all variables wrt loss
    optimizer.zero_grad()
    loss.backward()

    # performs updates using calculated gradients
    optimizer.step()

    # Evaluate summaries only once in a while
    if i % SAVE_SUMMARY_STEPS == 0:
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    return loss


def train_triplets(model, optimizer, loss_fn, metrics, train_batch, labels_batch, summ, i):

    # Extract triplets
    train_batch_anchor, train_batch_positive, train_batch_negative = train_batch

    # move to GPU if available
    if params.cuda:
        train_batch_anchor = train_batch_anchor.cuda()
        train_batch_positive = train_batch_positive.cuda()
        train_batch_negative = train_batch_negative.cuda()
        labels_batch = labels_batch.cuda()

    # convert to torch Variables
    train_batch_anchor = Variable(train_batch_anchor)
    train_batch_positive = Variable(train_batch_positive)
    train_batch_negative = Variable(train_batch_negative)
    labels_batch = Variable(labels_batch)

    # compute model output and loss
    output_batch_anchor = model(train_batch_anchor)
    output_batch_positive = model(train_batch_positive)
    output_batch_negative = model(train_batch_negative)
    loss = loss_fn(output_batch_anchor, output_batch_positive, output_batch_negative)

    optimizer.zero_grad()
    loss.backward()

    # performs updates using calculated gradients
    optimizer.step()

    # Evaluate summaries only once in a while
    if i % SAVE_SUMMARY_STEPS == 0:
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch_anchor.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    return loss


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            if params.loss == 'triplet':
                loss = train_triplets(model, optimizer, loss_fn, metrics, train_batch, labels_batch, summ, i)
            else:
                loss = train_other(model, optimizer, loss_fn, metrics, train_batch, labels_batch, summ, i)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean, loss_avg()


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(params.num_epochs):
        # Step in scheduler
        scheduler.step()
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics, train_loss = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        if params.loss == 'triplet':
            val_metrics, val_loss = evaluate_triplets(model, loss_fn, val_dataloader, metrics, params)
        else:
            val_metrics, val_loss = evaluate_crossentropy(model, loss_fn, val_dataloader, metrics, params)

        # Append to lists to keep trak of results in every epoch
        train_accuracies.append(train_metrics['accuracy'])
        train_losses.append(train_loss)
        val_accuracies.append(val_metrics['accuracy'])
        val_losses.append(val_loss)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    # Save all accuracies and losses
    all_json_path = os.path.join(model_dir, "acc_loss.p")
    utils.save_dict_to_pickle(val_metrics, all_json_path)


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

    logging.info("- done.")

    # Define the model and optimizer
    choices = {
        'linear': LinearRegression().cuda() if params.cuda else LinearRegression(),
        'mlp': MLP().cuda() if params.cuda else MLP(),
        'cnn': CNN().cuda() if params.cuda else CNN()
    }
    model = choices[args.model]
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    # fetch loss function and metrics
    losses = {
        'crossentropy': torch.nn.CrossEntropyLoss(),
        'hinge': torch.nn.MultiMarginLoss(),
        'mse': torch.nn.MSELoss(),
        'triplet': torch.nn.TripletMarginLoss()
    }
    loss_fn = losses[args.loss]
    params.loss = args.loss
    metrics = dict_metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)