## Training code for the Imitation learning model
## Use the Driving simulator attached to this code to drive a car
## Drive nicely as the Imitation model will try to Imitate you.
## After drving, close the simulator. It will save a log of your driving data
## The training code will access this data for training
## For the model to perform well, one needs to drive 
# atleast for more than an hour.
## Attaching a sample training dataset, so that one can train directly 
#  without having to generate the expert training data
## The training will save a model in the folder new_training_model

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import scipy
from scipy import signal

# ## Helper functions
def toDevice(datas, device):
    """Enable cuda."""
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)


def augment(img, angle):
    """Data augmentation."""
    #name = dataroot 
    current_image = cv2.imread(img)
    #cropping image to remove the sky
    current_image = current_image[60::,::]
    if np.random.rand() < 0.5:
        # data augmentation
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0
    
    return current_image, angle


def change_bright(img):
    # convert rgb to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = np.random.uniform(0.5,1.0)
    # increase of brightness in random images
    hsv[:,:,2] = rand*hsv[:,:2]
    # convert back to rgb
    new_img = cv2.cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return new_img


def load_data(img_paths,steers, test_size):
    """Load training data and train validation split"""

    # Divide the data into training set and validation set
    data_df = pd.DataFrame({'center':img_paths,'steering':steers})
    train_len = int(test_size * len(steers))
    valid_len = len(steers) - train_len
    trainset, valset = data.random_split(
        data_df.values.tolist(), lengths=[train_len, valid_len])

    return trainset, valset

## Create dataset
class TripletDataset(data.Dataset):

    def __init__(self,dataroot,samples, transform=None):
        self.samples = samples
        self.dataroot = dataroot
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[1])
        center_img, steering_angle_center = augment(batch_samples[0],steering_angle)
        return (center_img, steering_angle_center)

    def __len__(self):
        return len(self.samples)


# ## Get data loader
def data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers):
    """dataset Loader.
    Args:
        trainset: training set
        valset: validation set
        batch size
        shuffle ratio
        num_workers: number of workers in DataLoader

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set
        testloader (torch.utils.data.DataLoader): DataLoader for validation set
    """
    transformations = transforms.Compose(
        [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

    # Load training data and validation data
    training_set = TripletDataset(dataroot,trainset, transformations)
    trainloader = DataLoader(training_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    validation_set = TripletDataset(dataroot, valset, transformations)
    valloader = DataLoader(validation_set,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

    return trainloader, valloader


# Define model
class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """The NVIDIA architecture.
            Data preprocessing and image normalisation
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            The 5 convolution layers are for feature extraction.
            The fully connected layers are predict the steering angless

        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            # convolution layer 1
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            # convolution layer 2
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            # convolution layer 3
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            # convolution layer 4
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            # convolution layer 5
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            # fully connected layer 1
            nn.Linear(in_features=64 * 2 * 425, out_features=150),
            nn.ELU(),
            # fully connected layer 2
            nn.Linear(in_features=150, out_features=80),
            nn.ELU(),
            # fully connected layer 3
            nn.Linear(in_features=80, out_features=10),
            # fully connected layer 4
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward propogation"""
        # change the tensor shape
        input = input.view(input.size(0), 3, 196, 455)
        # pass the input to the 5 convolution layers
        output = self.conv_layers(input)
        # reshape the features to pass it into activation layers
        output = output.view(output.size(0), -1)
        # pass the feature vectors to the fully connected layers
        output = self.linear_layers(output)
        return output

class Trainer(object):
    """Trainer."""

    def __init__(self,ckptroot,model,device,epochs,criterion,optimizer,scheduler,start_epoch,trainloader,validationloader):
        """Self-Driving car Trainer.
        Args:
            model:
            device:
            epochs:
            criterion:
            optimizer:
            start_epoch:
            trainloader:
            validationloader:

        """
        super(Trainer, self).__init__()

        self.model = model
        self.device = device
        self.epochs = epochs
        self.ckptroot = ckptroot
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.trainloader = trainloader
        self.validationloader = validationloader

    def train(self):
        """Training process."""
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.scheduler.step()
            
            # Training
            train_loss = 0.0
            train_acc = 0.0
            self.model.train()

            for local_batch, (centers) in enumerate(self.trainloader):
                # Transfer to GPU
                centers= toDevice(centers, self.device)
                # Model computations
                self.optimizer.zero_grad()
                datas = [centers]
                
                for data in datas:
                    imgs, angles = data
                    outputs = self.model(imgs)
                    #print (outputs)
                    loss = self.criterion(outputs, angles.unsqueeze(1))
                    correct = (angles.eq(outputs.long())).sum()
                    # back propagation
                    loss.backward()
                    self.optimizer.step()
                    #calculate the traimning loss
                    train_loss += loss.data.item()
                    train_acc += correct.data.item()
                    

                if local_batch % 500 == 0:

                    print("Training Epoch: {} | Loss: {}".format(epoch, train_loss / (local_batch + 1)))
                    print("Training Accuracy: {} | Acc: {}".format(epoch, (train_acc / (local_batch + 1))))
            # Validation
            self.model.eval()
            valid_loss = 0
            valid_acc = 0
            with torch.set_grad_enabled(False):
                for local_batch, (centers) in enumerate(self.validationloader):
                    # Transfer to GPU
                    centers = toDevice(centers, self.device)

                    # Model computations
                    self.optimizer.zero_grad()
                    datas = [centers]
                    for data in datas:
                        # extract the dataloader
                        imgs, angles = data
                        # model prediction
                        outputs = self.model(imgs)
                        # calculating the model loss
                        loss = self.criterion(outputs, angles.unsqueeze(1))
                        correct = (angles.eq(outputs.long())).sum()

                        valid_loss += loss.data.item()
                        valid_acc += correct.data.item()
                    if local_batch % 500 == 0:
                        print("Validation Loss: {}".format(valid_loss / (local_batch + 1)))
                        print("Accuracy: {}".format((valid_acc / (local_batch + 1))))

            print()
            # Save model
            if epoch % 5 == 0 or epoch == self.epochs + self.start_epoch - 1:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save_checkpoint(state)

    def save_checkpoint(self, state):
        """Save checkpoint."""
        print("==> Save checkpoint ...")
        if not os.path.exists(self.ckptroot):
            os.makedirs(self.ckptroot)
        # save the trained model
        torch.save(state, self.ckptroot + 'real.ckpt')


def main():
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    ## Load the data you want ot test
    data_path = "training_set.txt"
    dataroot = []
    steers = []
    ## data loading
    with open(data_path) as file:
        for line in file:
            if line.split(',')[0]=="center":continue
            dataroot.append('driving_dataset/' + line.split(' ')[0])
            steers.append(line.split(' ')[1].strip())
    # model hyper parameters
    lr = 1e-5
    weight_decay = 1e-5
    batch_size = 5 ##32
    num_workers = 8
    test_size = 0.8
    shuffle = True
    epochs = 1 #80
    start_epoch = 0
    save = 'new_training_model/'
    trainset, valset = load_data(dataroot,steers, test_size)
    trainloader, validationloader = data_loader(dataroot,
                                            trainset, valset,
                                            batch_size,
                                            shuffle,
                                            num_workers)
    # Define model
    print("==> Initialize model ...")
    model = NetworkNvidia()
    print("==> Initialize model done ...")
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay)
    # Define the loss function
    criterion = nn.MSELoss()
    # learning rate scheduler   
    scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    # Use Gpu if available..else run CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Accesing you device CPU/GPU?(cuda)",device)
    print("==> Start training ...")
    # starting the imitation learning training
    trainer = Trainer(save,
                    model,
                    device,
                    epochs,
                    criterion,
                    optimizer,
                    scheduler,
                    start_epoch,
                    trainloader,
                    validationloader)
    trainer.train()

if __name__ == "__main__":
    main()

