"""Evaluates the model"""

import argparse
import logging
import os

import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import utils
from metrics import dict_metrics
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate_triplets(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # Last layer activation
    activation = torch.nn.Softmax(dim=1)

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # Extract triplets
        train_batch_anchor, train_batch_positive, train_batch_negative = data_batch

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

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = activation(output_batch_anchor).data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, loss


def evaluate_crossentropy(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # Last layer activation
    activation = torch.nn.Softmax(dim=1)

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = activation(output_batch).data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, loss


def predict(model, dataloader, params, category_function):
    """Predict the category on test.
    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        category_function: (function) function to return number -> category
    """

    # set model to evaluation mode
    model.eval()

    # Last layer activation
    activation = torch.nn.Softmax()

    # summary for current eval loop
    predictions = []
    indices = []

    # compute metrics over the dataset
    for data_batch, idx_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, idx_batch = data_batch.cuda(), idx_batch.cuda()
        # fetch the next evaluation batch
        data_batch, idx_batch = Variable(data_batch), Variable(idx_batch)

        # compute model output
        output_batch = model(data_batch)
        output_batch = activation(output_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = np.argmax(output_batch.data.cpu().numpy(), axis=1)
        idx_batch = idx_batch.data.cpu().numpy()

        predictions.append(output_batch)
        indices.append(idx_batch)

    predictions = np.apply_along_axis(category_function, 0, np.concatenate(predictions))
    indices = np.concatenate(indices)

    return {'Id': indices, 'Category': predictions}


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Hyperparameters(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = dict_metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Predict
    inverse = np.load('inverse.npy').item()
    predictions = predict(model, test_dl, params, np.vectorize(lambda x: inverse[x]))

    # Create csv
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv('results.csv', index=False)