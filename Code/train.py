import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import model
import datasets
import copy
import numpy as np
import random
from matplotlib import pyplot as plt
from torchsummary import summary




EPOCH_SIZE = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY_VALUE = 2e-5
BATCH_SIZE = 4




# Make sure to use the GPU. The following line is just a check to see if GPU is availables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# load your dataset and dataloader
# feel free to change header of bird_dataset class
root = 'birds_dataset/'
train_dataset = datasets.bird_dataset(root,'train_list.txt')
test_dataset = datasets.bird_dataset(root,'test_list.txt')

valid_dataset = datasets.bird_dataset(root,'train_list.txt')

valid_indices = []

NUM_EXAMPLES_PER = 30
for i in range(0, len(train_dataset), NUM_EXAMPLES_PER):
    valid_indices += random.sample(range(i, i + NUM_EXAMPLES_PER), 3)

train_dataset.images = [train_dataset.images[i] for i in range(len(train_dataset.images)) if i not in valid_indices]
train_dataset.labels = [train_dataset.labels[i] for i in range(len(train_dataset.labels)) if i not in valid_indices]

valid_dataset.images = [valid_dataset.images[i] for i in range(len(valid_dataset.images)) if i in valid_indices]
valid_dataset.labels = [valid_dataset.labels[i] for i in range(len(valid_dataset.labels)) if i in valid_indices]


# Fill in optional arguments to the dataloader as you need it
train_dataloader1 = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

valid_dataloader1 = DataLoader(valid_dataset, batch_size=BATCH_SIZE)



# CREATE SECOND FOLD
train_dataset = datasets.bird_dataset(root,'train_list.txt')

valid_dataset = datasets.bird_dataset(root,'train_list.txt')

valid_indices = []

NUM_EXAMPLES_PER = 30
for i in range(0, len(train_dataset), NUM_EXAMPLES_PER):
    valid_indices += random.sample(range(i, i + NUM_EXAMPLES_PER), 2)

train_dataset.images = [train_dataset.images[i] for i in range(len(train_dataset.images)) if i not in valid_indices]
train_dataset.labels = [train_dataset.labels[i] for i in range(len(train_dataset.labels)) if i not in valid_indices]

valid_dataset.images = [valid_dataset.images[i] for i in range(len(valid_dataset.images)) if i in valid_indices]
valid_dataset.labels = [valid_dataset.labels[i] for i in range(len(valid_dataset.labels)) if i in valid_indices]

train_dataloader2 = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataloader2 = DataLoader(valid_dataset, batch_size=BATCH_SIZE)




# train your model
# For each epoch iterate over your dataloaders/datasets, pass it to your NN model, get output, calculate loss and
# backpropagate using optimizer
def train(train_dataloader, valid_dataloader, best_loss, best_model):
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    
    for epoch in range(EPOCH_SIZE):
        print("Epoch:", epoch)
        epoch_loss = 0
        accurate = 0
        
        nn_model.train()
        for i, (image, label) in enumerate(train_dataloader):
            image = image.cuda().float()
            label = label.cuda()

            output = nn_model.forward(image)

            loss = criterion(output, label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value, index = torch.max(output.data, 1)
            accurate += (index == label).sum().item()

#             if index.item() == label.item():
#                 accurate += 1

        epoch_loss = epoch_loss/len(train_dataloader.dataset)
        accuracy = accurate/len(train_dataloader.dataset)
        print("Train Accuracy:", accuracy, "| Loss:", epoch_loss)
        train_loss.append(epoch_loss)
        train_acc.append(accuracy)

        nn_model.eval()
        epoch_loss = 0
        accurate = 0
        with torch.no_grad():
            for i, (image, label) in enumerate(valid_dataloader):
                image = image.cuda().float()
                label = label.cuda()

                output = nn_model.forward(image)

                loss = criterion(output, label)
                epoch_loss += loss.item()

                value, index = torch.max(output.data, 1)
                accurate += (index == label).sum().item()
#                 if index.item() == label.item():
#                     accurate += 1

        epoch_loss = epoch_loss/len(valid_dataloader.dataset)
        accuracy = accurate/len(valid_dataloader.dataset)
        print("Valid Accuracy:", accuracy, "| Loss:", epoch_loss)
        valid_loss.append(epoch_loss)
        valid_acc.append(accuracy)

        if (epoch_loss < best_loss):
            best_loss = epoch_loss
            best_model = copy.deepcopy(nn_model)
    
    return best_model, train_loss, valid_loss, train_acc, valid_acc



best_loss = float('inf')
best_model = None

# Create NN model object
nn_model = model.baseline_Net(classes = 20)
nn_model.to(device)


weight_initialization = [nn.Conv2d, nn.Linear]

for layer in nn_model.modules():
    if type(layer) in weight_initialization:
    #if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)


# Create loss functions, optimizers
# For baseline model use this
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_VALUE)
# Initialize weights

best_model, train_loss1, valid_loss1, train_acc1, valid_acc1 = train(train_dataloader1, valid_dataloader1, best_loss, best_model)


# PLOT LOSS/ACCURACY


# SECOND FOLD
# Create NN model object
nn_model = model.baseline_Net(classes = 20)
nn_model.to(device)


weight_initialization = [nn.Conv2d, nn.Linear]

for layer in nn_model.modules():
    if type(layer) in weight_initialization:
    #if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)


# Create loss functions, optimizers
# For baseline model use this
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_VALUE)
# Initialize weights

best_model, train_loss2, valid_loss2, train_acc2, valid_acc2 = train(train_dataloader2, valid_dataloader2, best_loss, best_model)

train_loss = [(train_loss1[i] + train_loss2[i])/2 for i in range(len(train_loss1))]
valid_loss = [(valid_loss1[i] + valid_loss2[i])/2 for i in range(len(valid_loss1))]

train_acc = [(train_acc1[i] + train_acc2[i])/2 for i in range(len(train_acc1))]
valid_acc = [(valid_acc1[i] + valid_acc2[i])/2 for i in range(len(valid_acc1))]



# PLOT LOSS/ACCURACY

plt.errorbar(range(len(train_loss)), train_loss, color = 'blue', label='train loss')
plt.errorbar(range(len(valid_loss)), valid_loss, color = 'red', label='valid loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title('Loss For Each Epoch')


plt.legend()
plt.show()

plt.errorbar(range(len(train_acc)), train_acc, color = 'blue', label='train accuracy')
plt.errorbar(range(len(valid_acc)), valid_acc, color = 'red', label='valid accuracy')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Accuracy For Each Epoch')

plt.legend()
plt.show()

test_loss = 0
test_accurate = 0

best_model.eval()
for i, (image, label) in enumerate(test_dataloader):
    image = image.cuda().float()
    label = label.cuda()

    output = best_model.forward(image)

    loss = criterion(output, label)
    test_loss += loss.item()

    value, index = torch.max(output.data, 1)
    test_accurate += (index == label).sum().item()
#     if index.item() == label.item():
#         test_accurate += 1

test_loss = test_loss/len(test_dataloader.dataset)
test_accuracy = test_accurate/len(test_dataloader.dataset)

print("Test Accuracy:", test_accuracy, "| Loss:", test_loss)
print("Epoch size:", EPOCH_SIZE, "Learning rate:", LEARNING_RATE)

print(summary(best_model, (3, 224, 224)))
print(best_model)
# Save your model/best model





# PLOT WEIGHT

# best_weights = np.delete(best_weights, 0, axis = 0)
# best_weights_transposed = np.transpose(best_weights)

# # Show Weights
# for i in range(len(best_weights_transposed)):
#     best_weights_reconstructed = projection(best_weights_transposed[i], np.transpose(PC))
#     x = plt.matshow(best_weights_reconstructed.reshape(28, 28), cmap='Oranges')
#     plt.colorbar(x)
#     plt.show()