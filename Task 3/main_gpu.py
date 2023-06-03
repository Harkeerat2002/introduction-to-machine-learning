# Libraries
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

X_train = torch.FloatTensor([0., 1., 2.])
X_train = X_train.to(device)
X_train.is_cuda

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        root="dataset/", transform=train_transforms)
    
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=False,
                          pin_memory=True, num_workers=16)

    
    model = resnet50(weights='DEFAULT', pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)

    embeddings = []
    embedding_size = 2048
    num_images = len(train_dataset)
    model.eval()
    
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = embeddings.reshape(num_images, embedding_size)

    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '')
                 for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')

    # Normalize the embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def create_loader_from_np(X, y=None, train=True, batch_size=64, shuffle=True, num_workers=4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self, embedding_size=2048):
        """
        The constructor of the model.

        :param embedding_size: the size of the image embedding.
        """
        super().__init__()
        # Creating new layers on top of the pretrained model
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3 * embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        The forward pass of the model.

        :param x: torch.Tensor, the input to the model.

        :return: torch.Tensor, the output of the model.
        """
        x = self.fc_layers(x)

        return x
    

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.to(device)
    number_of_epochs = 20

    criterion = nn.BCELoss().to(device) # Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Optimizer

    random_generator = torch.Generator(device=device)
    train_data, valid_data = random_split(train_loader.dataset, [0.8, 0.2], generator=random_generator)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    # Training
    for epoch in range(number_of_epochs):
        epoch = epoch + 1
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.type_as(output))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target.type_as(output))
                valid_loss += loss.item()
        valid_loss /= len(valid_loader.dataset)
        

        # Taken from college
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():  # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')

# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if (os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    print("Generated embeddings")

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    print("Loaded data")

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train=True, batch_size=64)
    test_loader = create_loader_from_np(
        X_test, train=False, batch_size=2048, shuffle=False)
    
    print("Created data loaders")

    # define a model and train it
    model = train_model(train_loader)

    print("Trained model")

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")