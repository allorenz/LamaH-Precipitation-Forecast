import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from preprocessing import run_preprocessing, load_preprocessed_data


class CustomDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
  '''

  def __init__(self, X, y):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.tensor(X_train.values, dtype=torch.float32)
      self.y = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]
  
class NeuralNet(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(24, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)


def train_model(model, num_epochs, dataloader):
    #num_epochs = 10
    for epoch in range(num_epochs):
        #model.train()
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            # Get and prepare inputs
            X, y_true = data

            # forward: predict
            y_pred = model(X)

            # compute loss
            loss = torch.sqrt(loss_function(y_pred, y_true))

            # backpropogating gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # Updating parameters(weights and bias)
            optimizer.step()

            current_loss += loss.item()

        print("Epoch {}, Training loss: {}".format(epoch, current_loss / len(dataloader)))

    return model


def evaluate_model(model, dataloader):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for data in dataloader:
            # Get and prepare inputs, move to device
            X, y_true = data

            # Forward pass: predict
            y_pred = model(X)

            # Compute loss
            loss = torch.sqrt(loss_function(y_pred, y_true))  # RMSE

            # Accumulate the loss
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Test set RMSE: {avg_loss:.4f}')

def split_data(df):
    X = df.drop("prec", axis=1)
    y = df["prec"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # load and split
    df = load_preprocessed_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # create data loader
    train_dataset = CustomDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    test_dataset = CustomDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    # create model
    model = NeuralNet()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train model
    model = train_model(model, 10, trainloader)

    # evaluate model
    evaluate_model(model, testloader)

    
