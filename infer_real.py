## Generates prediction of the steering angles for a given dashboard image
## Calls the imitation learning model trained 
#  and obtained from the training code and imitates the expert's training data
## Access the test images from the testset folder to generate 
#  predictions on them

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from scipy import signal
import glob




def toDevice(datas, device):
    """Enable cuda."""
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)


def augment(img, angle):
    """Data augmentation."""
    #load the image
    current_image = cv2.imread(img)
    #cropping image to remove the sky
    current_image = current_image[60::,::]
    return current_image, angle


def change_bright(img):
    # convert rgb to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = np.random.uniform(0.5,1.0)
    # change the brightness value
    hsv[:,:,2] = rand*hsv[:,:2]
    # covert back hsv to rgb
    new_img = cv2.cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return new_img


def load_data(img_paths,steers, test_size):
    """Load training data and train validation split"""
    # Divide the data into training set and validation set
    data_df = pd.DataFrame({'center':img_paths,'steering':steers})
    train_set = "Null"
    valset = data_df.values.tolist()
    return train_set,valset



class TripletDataset(data.Dataset):
    # Pytorch standard data load 
    def __init__(self,dataroot,samples, transform=None):
        self.samples = samples
        self.dataroot = dataroot
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[1])
        # Data preprocessing
        center_img, steering_angle_center = augment(batch_samples[0],steering_angle)
        return (center_img, steering_angle_center)

    def __len__(self):
        return len(self.samples)


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



class Inference(object):
    """Testing"""

    def __init__(self,model,device,criterion,optimizer,validationloader,valset):
        super(Inference, self).__init__()
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.validationloader = validationloader
        self.sum = 0
        self.valset = valset
        self.imgList = []

    def test(self):
        """ inference"""
        self.model.to(self.device)
        self.model.eval()
        with torch.set_grad_enabled(False):
            for local_batch, (centers) in enumerate(self.validationloader):
                # Transfer to GPU
                centers = toDevice(centers, self.device)
                # Model computations
                self.optimizer.zero_grad()
                datas = [centers]
                for data in datas:
                    imgs, angles = data
                    # prediction from the  model
                    outputs = self.model(imgs)
                    print ("steering_angle=  ", outputs.tolist()[0], "true_angle =  ", angles[0])
                    print ("deviation =", angles[0]-outputs.tolist()[0][0])
                    print ()
                #print(self.valset[local_batch][0])
                file = glob.glob(self.valset[local_batch][0])
                test_img = cv2.imread(file[0])
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                org = (10, 50)
                # fontScale 
                fontScale = 0.5
                # Blue color in BGR 
                color = (255, 0, 0) 
                # Line thickness of 2 px 
                thickness = 1
                label = "True_steering_angle=  "+ str(round(abs(angles[0].item()),4))
                test_img = cv2.putText(test_img, label, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
                model_res = "Predicted_steering_angle=  "+ str(round(abs(outputs.tolist()[0][0]),4))
                test_img = cv2.putText(test_img, model_res, (10,30), font,  
                   fontScale, (0,255,0), thickness, cv2.LINE_AA) 
                h,w,_ = test_img.shape
                size = (w,h)
                self.imgList.append(test_img)

            out =cv2.VideoWriter("imitation.avi",cv2.VideoWriter_fourcc(*'DIVX'),15,size)
            for i in range(len(self.imgList)):
                out.write(self.imgList[i])
            out.release()      


def main():
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    ## Load the data you want ot test
    data_path = "test.txt"
    dataroot = []
    steers = []
    ## data loading
    with open(data_path) as file:
        for line in file:
            if line.split(',')[0]=="center":continue
            dataroot.append('testset/' + line.split(' ')[0])
            steers.append(line.split(' ')[1].strip())

    # Model hyperparameters 
    lr = 1e-5
    weight_decay = 1e-5
    batch_size = 1
    num_workers = 8
    test_size = 0.01
    shuffle = False
    # Load the data in from of Tensors
    trainset, valset = load_data(dataroot,steers, test_size)
    _, validationloader = data_loader(dataroot,
                                            trainset, valset,
                                            batch_size,
                                            shuffle,
                                            num_workers)
    
    # Call the network
    print("Imitation model initialisation")
    model = NetworkNvidia()
    print("==> Initialize model done ...")
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay)
    # Define the loss function
    criterion = nn.MSELoss()
    # Use Gpu if available..else run CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Accesing you device CPU/GPU?(cuda)",device)
    # load the trained model 
    model = torch.load("imitation_model/real.ckpt")
    # Test the model output
    infer = Inference(model,
                    device,
                    criterion,
                    optimizer,
                    validationloader,valset)
    infer.test()
    

if __name__ == "__main__":
    main()

