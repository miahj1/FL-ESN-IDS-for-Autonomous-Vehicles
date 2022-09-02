import torch
import syft as sy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

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
        self.car_hacking_frame = pd.read_csv(csv_file, names=['Timestamp', 'CAN_ID', 'DLC', 'D0', 
                                                              'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                                                              'D7', 'Flag'])
        self.root_dir = root_dir
        self.transform = transform
        self.timestamps = self.car_hacking_frame['Timestamp']
        self.data = self.car_hacking_frame.loc[:, ['D0', 'D1', 'D2', 'D3',
                                                   'D4', 'D5', 'D6', 'D7']]
        self.can_id = self.car_hacking_frame['CAN_ID']
        self.flag = self.car_hacking_frame['Flag']

    def __getitem__(self,idx):
        '''Grabs relevant features from the dataset.'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ts = str(self.timestamps.iloc[idx])
        cid = str(self.can_id.iloc[idx])
        data = str(self.data.iloc[idx])
        flag = str(self.flag[idx])

        le = preprocessing.LabelEncoder()
        targets = le.fit_transform((ts, cid, data, flag))
        targets = torch.as_tensor(targets)

        return targets
    
    def attack_type(self, csv_file):
        '''
        Returns the number of packets that are 
        counted as T where T stands for injected.
        '''
        if "DoS" in csv_file:
            (self.flag=='T').sum()
        elif "Fuzzy" in csv_file:
            (self.flag=='T').sum()
        elif "gear" in csv_file:
            (self.flag=='T').sum()
        elif "RPM" in csv_file:
            (self.flag=='T').sum()
        else:
            print("The file does not contain any information about it's attack type!")
        
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

        le = preprocessing.LabelEncoder()
        targets = le.fit_transform((ts, cid, data))
        targets = torch.as_tensor(targets)

        return targets

    def __len__(self):
        return len(self.clean_validation_frame)

train_dataset = CarHackingDataset(csv_file='/content/car_hacking_data/DoS_dataset.csv', 
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
