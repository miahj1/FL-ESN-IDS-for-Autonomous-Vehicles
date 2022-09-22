import torch
import syft as sy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.optim as optim
from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir.util import NRMSELoss
import time

class CarHackingDataset(Dataset):
    """
    Loading the Car-Hacking Dataset from
    https://ocslab.hksecurity.net/Datasets/car-hacking-dataset 

    Args:
        csv_file: A path to the dataset file which has the extension CSV.
        root_dir: The directory of the parent folder of the dataset.
        transform (callable, optional): Optional tansform to be applied on a sample.
    """
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.car_hacking_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self,idx):
        '''Grabs relevant features from the dataset.'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = ['Timestamp', 'DLC', 'CAN_ID', 'Data']
        X_train = self.car_hacking_frame.loc[:, features].values
        X_train = torch.as_tensor(X_train)

        # It looks like it's a bad idea to encode features.
        # https://stackoverflow.com/questions/61217713/labelencoder-for-categorical-features

        return X_train[idx], self.car_hacking_frame['Flag'].iloc[idx]
            
    def __len__(self):
        return len(self.car_hacking_frame)


class ValidationDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str):
        self.validation_frame = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        '''Grabs relevant features from the dataset.'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = ['Timestamp', 'DLC', 'CAN_ID', 'Data']
        X_train = self.validation_frame.loc[:, features].values
        X_train = torch.as_tensor(X_train)

        return X_train[idx]

    def __len__(self):
        return len(self.validation_frame)

train_dataset = CarHackingDataset(csv_file='/content/car_hacking_data/balanced_fuzzy_dataset.csv', 
                                  root_dir='/content/car_hacking_data')

test_dataset = ValidationDataset(csv_file='/content/car_hacking_data/clean_scaled_normal_run.csv', 
                                 root_dir='/content/car_hacking_data')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hook = sy.TorchHook(torch)

car1 = sy.VirtualWorker(hook, id="car1")
car2 = sy.VirtualWorker(hook, id="car2")

args = {
    'batch_size' : 13084,
    'epochs' : 1
}

federated_train_loader = sy.FederatedDataLoader(train_dataset.federate((car1, car2)),
                                                batch_size=args['batch_size'], shuffle=True, drop_last=True)

test_loader = DataLoader(dataset=train_dataset,
                                 batch_size=args['batch_size'],
                                 drop_last=True,
                                 shuffle=True)

# Intializing the loss function which is probably a variation of mean squared error.
# This loss function may not be good for classification.
nrmse = NRMSELoss()

def train(model, device, federated_train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model = model.send(data.location)

        data, target = data.reshape(args['batch_size'], 4, -1).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = nrmse(output, target)
        loss.backward()
        optimizer.step()
        model.get()

        if batch_idx % 10 == 0:
            loss = loss.get()

            print(f'''Train Epoch: {epoch} [{(batch_idx * args['batch_size'])}/{(len(federated_train_loader) * args['batch_size'])}'''
                   + f'''({100. * batch_idx / len(federated_train_loader):.0f}%)]\tLoss: {loss.item():.6f}''')

def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct/len(test_loader.dataset)))

model = DeepESN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

t0 = time.time()
for epoch in range(1, args['batch_size'] + 1):
    train(model, device, federated_train_loader, optimizer, epoch)
    test(model, device, test_loader)
t1 = time.time()
print(f'Training took {t1 - t0}s')