import os
import torch
from pathlib import Path
from torch import nn
from sklearn.model_selection import train_test_split
from src.script_files.preprocessing import run_preprocessing, load_preprocessed_data
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent


class CustomDataset(torch.utils.data.Dataset):
  '''
  Prepare the Custom dataset for regression
  '''

  def __init__(self, X, y):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      self.X = torch.tensor(X.values, dtype=torch.float32)
      self.y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

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
      nn.Linear(25, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    return self.layers(x)
  
  
  def fit(self, num_epochs, trainloader, testloader, checkpoint_dir=PROJECT_ROOT/"model"/"checkpoints"): # "../model/checkpoints"
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    # Early stopping parameters
    patience = 5 # Number of epochs to wait before stopping
    min_improvement = 0.01 # Minimum change in loss to qualify as an improvement
    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        self.train()
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0

        for i, data in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}', leave=False), 0):
            # prepare
            X, y_true = data

            # forward: predict
            y_pred = self.forward(X)

            # compute loss
            loss = torch.sqrt(loss_function(y_pred, y_true))

            # backpropagating gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # Updating parameters (weights and bias)
            optimizer.step()

            current_loss += loss.item()

        print("Epoch {}, Training loss: {}".format(epoch, current_loss / len(trainloader)))

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                X, y_true = data
                y_pred = self.forward(X)
                loss = torch.sqrt(loss_function(y_pred, y_true))  # RMSE
                val_loss += loss.item()
                val_steps += 1
        print("Epoch {}, Validation loss: {}".format(epoch, val_loss / val_steps))

        # save model
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
        torch.save(self.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Early Stopping - Check for improvement
        if val_loss < best_loss - min_improvement:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    

  def predict(self, X_test):
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32)
            predictions = self.forward(X_test)
        return predictions.numpy()


  def evaluate(self, dataloader):
      loss_function = nn.MSELoss()
      with torch.no_grad():
          self.eval()
          total_loss = 0.0
          for data in dataloader:
              # Get and prepare inputs
              X, y_true = data

              # Forward pass: predict
              y_pred = self.forward(X)

              # Compute loss
              loss = torch.sqrt(loss_function(y_pred, y_true))  # RMSE

              # Accumulate the loss
              total_loss += loss.item()

          avg_loss = total_loss / len(dataloader)
          print(f'Test Set RMSE: {avg_loss:.4f}')
      path = PROJECT_ROOT / "output" / "neural_net_result.txt"
      with open(path, "w") as file:
         file.write(f'Test Set RMSE: {avg_loss:.4f}mm.')
         


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
    
    # train model
    model.fit(num_epochs=25, trainloader=trainloader, testloader=testloader)

    # evaluate model
    model.evaluate(dataloader=testloader)