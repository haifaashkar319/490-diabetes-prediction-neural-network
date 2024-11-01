import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class DiabetesModel:
    def __init__(self, X, y, hidden_size=64, learning_rate=0.001, test_size=0.2, val_size=0.25, batch_size=32):
        input_size = X.shape[1]

        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Set batch size
        self.batch_size = batch_size

        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs=100):
        # Prepare DataLoader for training and validation data
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        val_dataset = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Store training and validation losses for visualization
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            # Average training loss for the epoch
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation phase
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    running_val_loss += loss.item() * inputs.size(0)

            # Average validation loss for the epoch
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}] '
                      f'Training Loss: {epoch_train_loss:.4f} '
                      f'Validation Loss: {epoch_val_loss:.4f}')

        # Plot training and validation losses and save the figure
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig('training_validation_loss_plot.png')  # Save plot as a file
        plt.show()
        print("Plot saved as 'training_validation_loss_plot.png'")

    def evaluate(self):
        # Evaluate on the test data
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test_tensor)
            test_loss = self.criterion(test_outputs, self.y_test_tensor).item()
            test_mse = mean_squared_error(self.y_test_tensor.numpy(), test_outputs.numpy())
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        
    def predict(self, X):
        # Set model to evaluation mode for prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions
