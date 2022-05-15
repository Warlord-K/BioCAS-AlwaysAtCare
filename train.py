from model import AudioClassifier, training, inference
import torch
import pandas as pd
from audio import SoundDS
from torch.utils.data import random_split


df = pd.read_csv('data_encoded_new.csv')
data_path = "./SPRSound/wav/"
myds = SoundDS(df, data_path)
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
try:
    myModel.load_state_dict(torch.load("./model_state_dict.pt"))
    print("Model Found")
except:
    print("No model found")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
print(next(myModel.parameters()).device)


num_epochs= 100
training(myModel, train_dl, num_epochs)
torch.save(myModel.state_dict(), "./model_state_dict.pt")
inference(myModel, val_dl)