import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random

MODEL_PARAM_FILE = "./.trained_model_parameters"


def get_data(path_to_data_file, train_mode=False):
    """
    Given a text file output of make data, parses each line into a label and 45 member feature vector.
    :param path_to_data_file:
    :param train_mode:
    :return:
    """
    # set seed for repeatability; maybe remove
    random.seed(2023)

    with open(path_to_data_file, "r") as data_file:
        lines = data_file.readlines()

    # To make sure that all of a single proteins examples aren't stuck together
    random.shuffle(lines)

    parsed_data = []
    if train_mode:
        # This tells the script to expect labels

        for line in lines:
            split_line = line.split()
            parsed_data.append( (int(split_line[0]), np.array(list(map(float, split_line[1:])))) )

        partition_ind = len(parsed_data) // 10
        train_data, test_data = parsed_data[:-partition_ind], parsed_data[-partition_ind:]
        return train_data, test_data

    else:
        # This tells the script to skip labels and just get feature vectors. Allows for batch prediction on multiple
        # proteins.
        for line in lines:
            split_line = line.split()
            parsed_data.append(np.array(list(map(float, split_line))))
        return parsed_data


class DisulfideBondSiteNN(nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: Ask prof about using bias; Im assuming I should
        self.lin1 = nn.Linear(45, 128, bias=True, dtype=float)
        self.lin2 = nn.Linear(128, 32, bias=True, dtype=float)
        self.lin3 = nn.Linear(32, 1, bias=True, dtype=float)

    def forward(self, distance_vector):
        h1 = torch.relu(self.lin1(distance_vector))
        h2 = torch.relu(self.lin2(h1))
        y_pred = torch.sigmoid_(self.lin3(h2))

        return y_pred


def loss_fn(pred_y, obv_y):
    return abs((pred_y - obv_y))


def train(model: nn.Module, path_to_training_data="feature_vectors.txt", num_epochs=1, learning_rate=.0001):
    data_train, data_test = get_data(path_to_training_data, train_mode=True)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_record = []
    loss_bucket = []

    for epoch in range(num_epochs):
        for data_pair in data_train:

            optimizer.zero_grad()
            pred_y = model(tensor(data_pair[1]))
            loss = loss_fn(pred_y, data_pair[0])
            loss_bucket.append(float(loss))
            # This is to help smooth out the loss function to get a better idea of the trend
            if len(loss_bucket) == 10:
                loss_record.append(sum(loss_bucket)/len(loss_bucket))
                loss_bucket = []
            loss.backward()
            optimizer.step()

    # evaluation step
    total = 0
    result_mat = np.zeros((2, 2))
    for data_pair in data_test:
        classification = round(float(model(tensor(data_pair[1]))))

        correct = classification == data_pair[0]
        positive = data_pair[0] == 1

        if correct and positive:
            result_mat[0, 0] += 1
        elif not correct and positive:
            result_mat[1, 0] += 1
        elif not correct and not positive:
            result_mat[0, 1] += 1
        elif correct and not positive:
            result_mat[1, 1] += 1

        total += 1

    # Display evaluation data
    print(result_mat/total)
    plt.plot(range(len(loss_record)), loss_record)
    plt.show()

    # Save model so we don't have to retrain.
    torch.save(model.state_dict(), MODEL_PARAM_FILE)


if __name__ == "__main__":
    test_nn = DisulfideBondSiteNN()
    train(test_nn, "feature_vectors.txt", num_epochs=10)