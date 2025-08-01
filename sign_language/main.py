import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
# from utils.MyDataset import MyDataset

# switching to GPU for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


# loading and cleaning the data
# train_df = pd.read_csv("data\sign_mnist_train.csv")
# valid_df = pd.read_csv("data\sign_mnist_valid.csv")

train_df = pd.read_csv("data\sign_mnist_train.csv")
valid_df = pd.read_csv("data\sign_mnist_valid.csv")

y_train = train_df.pop('label')
y_valid = valid_df.pop('label')

x_train = train_df.values
x_valid = valid_df.values


x_train = train_df.values / 255
x_valid = valid_df.values / 255



class MyDataset(Dataset):
    def __init__(self, x_df, y_df):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)

BATCH_SIZE = 32
train_data = MyDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)


valid_data = MyDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N = len(valid_loader.dataset)



# creating the model

input_size = 28 * 28
n_classes = 26

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
)

model = torch.compile(model.to(device))

# training the model
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


epochs = 20
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

#save a new model
import joblib
joblib.dump(model, 'sign_language.joblib')