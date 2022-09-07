import torch
import syft as sy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from auto_esn.esn.esn import ESNBase
import torch.nn.functional as F
import torch.optim as optim
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.util import NRMSELoss

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
        self.car_hacking_frame = pd.read_csv(csv_file)[:10000]
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self,idx):
        '''Grabs relevant features from the dataset.'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = ['Timestamp', 'DLC', 'CAN_ID', 'Data']
        X_train = self.car_hacking_frame.loc[:, features].values
        X_train_reduced = StandardScaler().fit_transform(X_train)
        X_train_reduced = torch.as_tensor(X_train_reduced)

        # It looks it's a bad idea to to encode features.
        # https://stackoverflow.com/questions/61217713/labelencoder-for-categorical-features

        class_le = LabelEncoder()
        target = class_le.fit_transform(self.car_hacking_frame['Flag'].values)
        target = torch.as_tensor(target)

        return X_train_reduced[idx], target[idx]
            
    def __len__(self):
        return len(self.car_hacking_frame)


class ValidationDataset(Dataset):
    def __init__(self, txt_file: str, root_dir: str):
        self.validation_frame = pd.read_csv(txt_file, sep='\s+', header=None,
                                    names = ['TS_Dupe', 'Timestamp', 'ID_Dupe', 
                                    'CAN_ID', 'Floating Point', 'DLC', 'DLC_Dupe',
                                    'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'])
        self.clean_validation_frame = self.validation_frame.drop(columns=['TS_Dupe', 
                                                                          'ID_Dupe', 
                                                                          'Floating Point', 
                                                                          'DLC', 
                                                                          'DLC_Dupe'])
        self.clean_validation_frame['Data'] = (self.clean_validation_frame['D0'].astype(str) + 
                                               self.clean_validation_frame['D1'].astype(str) + 
                                               self.clean_validation_frame['D3'].astype(str) + 
                                               self.clean_validation_frame['D4'].astype(str) +
                                               self.clean_validation_frame['D5'].astype(str) + 
                                               self.clean_validation_frame['D6'].astype(str) +
                                               self.clean_validation_frame['D7'].astype(str))

    def __getitem__(self, idx):
        '''Grabs relevant features from the dataset.'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ts = str(self.clean_validation_frame['Timestamp'].iloc[idx])
        cid = str(self.clean_validation_frame['CAN_ID'].iloc[idx])
        data = self.clean_validation_frame['Data'].iloc[idx]

        le = LabelEncoder()
        targets = le.fit_transform((ts, cid, data))
        targets = torch.as_tensor(targets)

        return targets

    def __len__(self):
        return len(self.clean_validation_frame)

train_dataset = CarHackingDataset(csv_file='/content/car_hacking_data/flt_cid_modded_fuzzy_dataset.csv', 
                                  root_dir='/content/car_hacking_data')


train_loader = DataLoader(dataset=train_dataset,
                                 batch_size=32,
                                 drop_last=True,
                                 shuffle=True)

test_dataset = ValidationDataset(txt_file='/content/car_hacking_data/normal_run_data.txt', 
                                 root_dir='/content/car_hacking_data')

test_loader = DataLoader(dataset=train_dataset,
                                 batch_size=32,
                                 drop_last=True,
                                 shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


hook = sy.TorchHook(torch)

car1 = sy.VirtualWorker(hook, id="car1")
car2 = sy.VirtualWorker(hook, id="car2")

federated_train_loader = sy.FederatedDataLoader(train_dataset.federate((car1, car2)),
                                                batch_size=32, shuffle=True)

# Intializing the loss function.
nrmse = NRMSELoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        model = model.send(data.location)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = nrmse(output, target)
        loss.backward()
        optimizer.step()
        model.get()

        if batch_idx % 10 == 0:
            loss = loss.get()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * 64,
                    len(train_loader) * 64,
                    100. * batch_idx / len(train_loader),
                    loss.item())
                 )

model = GroupedDeepESN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, 1+1):
    train(model, device, federated_train_loader, optimizer, epoch)
