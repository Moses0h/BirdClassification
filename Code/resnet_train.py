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
import torchvision.models as models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def train():
    EPOCH_SIZE = 1
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY_VALUE = 2e-4
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

    # Create custom resnet model object
    weight_initialization = [nn.Conv2d, nn.Linear]
    nn_model = models.resnet18(pretrained=True)


    
    #FINETUNING------------------
    set_parameter_requires_grad(nn_model, True)
    count = 0
    #if you want to unfreeze specific layers
#     for layer in nn_model.modules():
#         if type(layer) in weight_initialization:
#             count += 1
#             if count >= 16 and count <= 19:
#                 layer.weight.requires_grad = True
# #                 layer.bias.requires_grad = True
    #----------------------------
    #set custom classifier with 20 categories
    nn_model.fc = nn.Linear(512, 20)
    nn_model.to(device)

    #only set weight if unfrozen
    for layer in nn_model.modules():
        if type(layer) in weight_initialization:
            if layer.weight.requires_grad: 
                torch.nn.init.xavier_uniform_(layer.weight)
                
    params_to_update = []
    for name,param in nn_model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print(name)

    criterion = nn.CrossEntropyLoss()

    #specify which layers to update weights for
    optimizer = torch.optim.Adam(params_to_update, lr=LEARNING_RATE)


    best_model, train_loss1, valid_loss1, train_acc1, valid_acc1 = train(train_dataloader1, valid_dataloader1, best_loss, best_model)




    # SECOND FOLD
    # Create NN model object
    nn_model = models.resnet18(pretrained=True)


    
    #FINETUNING------------------
    set_parameter_requires_grad(nn_model, True)
    count = 0
    #if you want to unfreeze specific layers
#     for layer in nn_model.modules():
#         if type(layer) in weight_initialization:
#             count += 1
#             if count >= 16 and count <= 19:
#                 layer.weight.requires_grad = True
# #                 layer.bias.requires_grad = True
    #----------------------------

    #set custom layer
    nn_model.fc = nn.Linear(512, 20)
    nn_model.to(device)

    #only set weight if unfrozen
    for layer in nn_model.modules():
        if type(layer) in weight_initialization:
            if layer.weight.requires_grad: 
                torch.nn.init.xavier_uniform_(layer.weight)
                
    #specify which layers to update weights for
    params_to_update = []
    for name,param in nn_model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)




    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, lr=LEARNING_RATE)

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


    test_loss = test_loss/len(test_dataloader.dataset)
    test_accuracy = test_accurate/len(test_dataloader.dataset)

    print("Test Accuracy:", test_accuracy, "| Loss:", test_loss)
    print("Epoch size:", EPOCH_SIZE, "Learning rate:", LEARNING_RATE)

    print(best_model)





    # First convolutional layer (for weight map)
    model_children = list(best_model.children())
    first_conv = best_model.conv1
#     first_conv = list(model_children[0].children())[0]
    first_conv.cpu()

    # Weight map
    for i, filter in enumerate(first_conv.weight):
        plt.subplot(8, 8, i+1) 
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()


    # Feature map
    image_dataset = datasets.bird_dataset(root,'test_list.txt')
    image_dataset.images = [image_dataset.images[0]]
    image_dataloader = DataLoader(image_dataset)

    first_image = None
    for i, (image, label) in enumerate(image_dataloader):
        if i == 0:
            first_image = image.float()
            break
            #idk how to get a single image from dataloader but you need dataloader to process it

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    #first conv layer
    best_model.conv1.register_forward_hook(get_activation('features'))
    best_model.cpu()
    output = best_model(first_image.cpu())

    act = activation['features'].squeeze()
    for idx in range(act.size(0)):
        plt.subplot(act.size(0)**0.5, act.size(0)**0.5, idx + 1)
        plt.imshow(act[idx])
        plt.axis('off')
    plt.show()
    
    ########################################
    best_model.layer2.register_forward_hook(get_activation('features'))
    best_model.cpu()
    output = best_model(first_image.cpu())

    #middle conv layer
    act = activation['features'].squeeze()
    for idx in range(act.size(0)):
        plt.subplot(act.size(0)**0.5, act.size(0)**0.5, idx + 1)
        plt.imshow(act[idx])
        plt.axis('off')
    plt.show()
    
    best_model.layer4.register_forward_hook(get_activation('features'))
    best_model.cpu()
    output = best_model(first_image.cpu())

    #last conv layer
    act = activation['features'].squeeze()
    for idx in range(act.size(0)):
        plt.subplot(act.size(0)**0.5+1, act.size(0)**0.5+1, idx + 1)
        plt.imshow(act[idx])
        plt.axis('off')
    plt.show()




