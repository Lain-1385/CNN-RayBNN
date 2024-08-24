import numpy as np
import torch 
from torch import nn, optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout1 = nn.Dropout(0.25) 

        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=20 * 6 * 6, out_features=512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        features = x
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x, features


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    fold_accuracies= []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []

    train_subset = datasets.CIFAR10(root="/home/lain1385/scratch/project/data_tao", transform=transform, train=True)
    val_subset = datasets.CIFAR10(root="/home/lain1385/scratch/project/data_tao", transform=transform, train=False)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)

    net = CNN((3, 32, 32), 10, 128)
    print(net)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch_train_loss = []
    epoch_train_accs = []
    epoch_test_loss = []
    epoch_test_accs = []
    test_accs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    n_ep=100

    for epoch in range(n_ep):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)   
            optimizer.zero_grad()
            outputs, features = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss.append(running_loss / len(train_loader))
        epoch_train_accs.append(100 * correct_train / total_train)
        train_loss = running_loss/len(train_loader)
        train_accuracy = correct_train/total_train
        print(f'train_dataset Epoch [{epoch+1}/{n_ep}], Loss: {train_loss:.5f}, Accuracy: {train_accuracy:.5f}')
        
        
        all_features = []

        correct = 0
        total = 0
        test_features = []
        test_labels = []
        predicted_labels = []
        true_labels=[]
        net.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, features = net(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                features = features.cpu().numpy()
                test_features.append(features)
                test_labels.append(labels.cpu().numpy())
                
                _, predicted = torch.max(outputs.data, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        val_loss = test_loss / len(val_loader)
        val_accuracy = correct_test / total_test
        print(f'val_dataset Epoch [{epoch+1}/{n_ep}], Loss: {val_loss:.5f}, Accuracy: {val_accuracy:.5f}')
        
    train_features = []
    train_labels = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            _, features = net(inputs)
            train_features.append(features.cpu().numpy())
            train_labels.append(labels.numpy())
    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)

    # Extract features from the validation set
    val_features = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            _, features = net(inputs)
            val_features.append(features.cpu().numpy())
            val_labels.append(labels.numpy())
    val_features = np.vstack(val_features)
    val_labels = np.concatenate(val_labels)
    
    accuracy = accuracy_score(val_labels, predicted_labels)  
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)
    print(f'result: Accuracy={accuracy:.5f}, Precision={precision:.5f}, Recall={recall:.5f}, F1 Score={f1:.5f}')
    print(f'Precision: {precision:.5f}, Recall: {recall:.5f}, F1 Score: {f1:.5f}')
    print(f'Epoch [{epoch+1}/{n_ep}], Test Loss: {test_loss/len(val_loader):.5f}, Test Accuracy: {100*correct_test/total_test:.5f}%')

    train_iters = range(len(epoch_train_accs))

    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    avg_f1_score = np.mean(fold_f1_scores)
    print(f'Average: Accuracy={avg_accuracy:.5f}, Precision={avg_precision:.5f}, Recall={avg_recall:.5f}, F1 Score={avg_f1_score:.5f}')
             
                
    outputs = outputs.cpu().numpy()

    return val_features, val_labels, train_features, train_labels

if __name__ == '__main__':
    

    val_features, val_labels, train_features, train_labels = main()
    
    np.save('val_features.npy',val_features)
    np.save('val_labels.npy',val_labels)
    np.save('train_features.npy',train_features)
    np.save('train_labels.npy',train_labels)
