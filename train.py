import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



batch_size=16
epoch=20
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'checkpoint/'
input_size=90
seq_len=43
n_hidden=128
fc_1=[128,32]
fc_2=[32,10]
n_stack=2
lr=1e-5
num_class=10





class EEGDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, EEG_filenames):      # txt_file 에 경로('EEG/EEG_filenames') + npy 파일명('EEG_filenames')이 모두 들어가야함

        #self.EEG_train=np.load(EEG_train)
        self.EEG_filenames = np.loadtxt(EEG_filenames, dtype='str')   # 실제 file_name '들' (리스트)


    def __getitem__(self, idx):
        name=self.EEG_filenames[idx]
        data = np.load(os.path.join('./train_data/EEG_datas', name))           # 해당 idx의 EMG_datas 파일을 열기
        sample = {'EEG_data': data}

        return sample


    def __len__(self):
        return len(self.EEG_filenames)





EEGDataset=EEGDataset(EEG_filenames='./train_data/EEG_filenames/EEG_filenames.txt')
dataloader = DataLoader(EEGDataset, batch_size=batch_size,shuffle=True, drop_last=True)       # drop_last=True !!!!!!!!!!!!!!!!!111



class EEG_LSTM(nn.Module):
    def __init__(self):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=n_hidden, num_layers=n_stack, batch_first=True)
        self.fc_layer1 = nn.Sequential(
            nn.Linear(fc_1[0], fc_1[1]),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(fc_2[0], fc_2[1]),
        )



    def forward(self, hidden_and_cell, x):
        #print('x.shape : {0}'.format(x.shape))          #torch.Size([16, 1, 128, 440])
        x=torch.squeeze(x,1)
        x = x.permute(0,2,1)        # transpose쓰면 안된다.
        #print('x.shape : {0}'.format(x.shape))           #torch.Size([16, 440, 128])
        outputs, hidden = self.lstm(x, hidden_and_cell)
        #print(type(hidden))    # <class 'tuple'>     (hn, cn) 으로 hidden_n과 cell_n이 들어있다.
        #print(len(hidden))      # 2
        #print(hidden[0].shape)   # hidden_n으로 torch.Size([4, 16, 512]) 이다.
        #print(hidden[0])   # hidden_n을 직접 보면 batch마다는 같고 layer마다는 다르다는 것을 확인할수 있음.
        #print('-'*100)
        #print(outputs.shape)
        print(outputs.shape)                               # torch.Size([16, 128, 250])
        output = outputs[:,-1,:]  # 최종 예측 output
        print(output.shape)                              # torch.Size([16, 96, 250])
        #output=torch.flatten(output,start_dim=1)
        #print(output.shape)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)
        print(output.shape)                                # torch.Size([16, 96, 2])
        output = output.permute(0, 2, 1)

        return output





model = EEG_LSTM()
model = model.cuda()
loss_func=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)




def train():
    loss_array = []
    hidden = torch.zeros(n_stack, batch_size, n_hidden, requires_grad=True)
    #hidden = torch.zeros(n_stack, batch_size, n_hidden*n_stack, requires_grad=True)
    hidden = hidden.to(device)
    cell = torch.zeros(n_stack, batch_size, n_hidden, requires_grad=True)
    #cell = torch.zeros(n_stack, batch_size, n_hidden*n_stack, requires_grad=True)
    cell = cell.to(device)
    for i in range(epoch):
        for j, data in enumerate(dataloader):
            #print(hidden[0])
            #print(hidden[0].shape)

            x = data['EEG_data']
            print(x.shape)
            y = data['EEG_data']
            print(y.shape)

            x=x[:,:90,:]
            print(x.shape)           # torch.Size([16, 64, 128])
            y=y[:,90,:]
            print(y.shape)           # torch.Size([16, 96])



            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            #print(x.shape)     (16,128,440)

            x = torch.unsqueeze(x, 1)
            #x = torch.unsqueeze(x, 1)

            x=x.to(device)
            y=y.to(device)
            #print(y)
            print(x.shape)            # torch.Size([16, 1, 64, 128])

            output = model.forward((hidden, cell),x)

            optimizer.zero_grad()
            #print(y)
            #print(y.shape)


            #output = output.reshape(-1,num_class)                  # num_class=40
            loss = loss_func(output, y)
            loss.backward()

            optimizer.step()
            #print(hidden[0])

            if j %500== 0:
                loss_array.append(loss)
                print(loss)
                #print(output)
                torch.save(model.state_dict(), PATH +'/'+'model{0}.pth'.format(i))

            if i*j > 1000000:
                break
        print("epoch : {0}".format(i))
        print("-"*100)





    plt.plot(loss_array)
    plt.show()





if __name__ == '__main__':
    train()
