import numpy as np
import raybnn_python
import torch 
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import os
from torchvision import datasets, transforms,utils
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torch import optim
import torch.nn.functional as F

def train_raybnn(x_train, y_train, x_test, y_test):
    accuracy_values = []
    precision_values=[]
    recall_values = []
    f1_values=[]
    if isinstance(x_train, torch.Tensor):
        Rey_train = x_train.cpu().numpy()

    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)

    dir_path = "/home/lain1385/scratch/project/CNN_with_RayBNN/temp_RayBNN"

    max_input_size = 720
    input_size = 720

    max_output_size = 10
    output_size = 10

    max_neuron_size = 1000

    batch_size = 1000
    traj_size = 1

    proc_num = 5
    active_size = 1000

    training_samples = int(x_train.shape[0]/batch_size)
    crossval_samples = training_samples
    testing_samples = int(x_test.shape[0]/batch_size)

    # Format MNIST dataset
    train_x = np.zeros((input_size, batch_size, traj_size, training_samples)).astype(np.float32)
    train_y = np.zeros((output_size, batch_size, traj_size, training_samples)).astype(np.float32)

    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i / batch_size)

        train_x[:, j, 0, k] = x_train[i, :]

        idx = y_train[i]
        train_y[idx, j, 0, k] = 1.0

    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)

    # Create Neural Network
    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,

        output_size,
        max_output_size,

        active_size,
        max_neuron_size,

        batch_size,
        traj_size,

        proc_num,
        dir_path
    )
    print("Create Neural Network success!")

    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]

    arch_search = raybnn_python.add_neuron_to_existing3(
        10,
        10000,
        sphere_rad / 1.3,
        sphere_rad / 1.3,
        sphere_rad / 1.3,

        arch_search,
    )

    arch_search = raybnn_python.select_forward_sphere(arch_search)

    raybnn_python.print_model_info(arch_search)

    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "SHUFFLE_CONNECTIONS"
    lr_strategy2 = "MAX_ALPHA"

    loss_function = "sigmoid_cross_entropy_5"

    max_epoch = 0
    stop_epoch = 100000
    stop_train_loss = 0.005

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

    total_epochs = 100

    for epoch in range(total_epochs):
        max_epoch += 1
        # Train Neural Network
        arch_search = raybnn_python.train_network(
            train_x,
            train_y,

            crossval_x,
            crossval_y,

            stop_strategy,
            lr_strategy,
            lr_strategy2,

            loss_function,

            max_epoch + 1,
            stop_epoch + 1,
            stop_train_loss,

            max_alpha,

            exit_counter_threshold,
            shuffle_counter_threshold,

            arch_search
        )

        test_x = np.zeros((input_size, batch_size, traj_size, testing_samples)).astype(np.float32)

        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            test_x[:, j, 0, k] = x_test[i, :]

        # Test Neural Network
        output_y = raybnn_python.test_network(
            test_x,

            arch_search
        )

        print(output_y.shape)

        pred = []
        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            sample = output_y[:, j, 0, k]

            pred.append(np.argmax(sample))

        pred = [np.argmax(output_y[:, i % batch_size, 0, int(i/batch_size)]) for i in range(x_test.shape[0])]
        acc = accuracy_score(y_test, pred)

        ret = precision_recall_fscore_support(y_test, pred, average='macro')

        print(acc)
        print(ret)

        accuracy_values.append(acc)
        precision_values.append(ret[0])
        recall_values.append(ret[1])
        f1_values.append(ret[2])

    print(output_y.shape)
    return output_y.reshape(-1)




if __name__ == '__main__':

    val_features = np.load('val_features.npy')
    val_labels = np.load('val_labels.npy')
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    
    output_y = train_raybnn(train_features, train_labels, val_features, val_labels)
