import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary


class FCNet(nn.Module):
    def __init__(self, input_size, num_classes: int, dropout_rate: float = 0.5):
        super(FCNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return self.softmax(x)


class FCNetModel:
    def __init__(self, num_classes: int, device, dropout_rate: float = 0.5, model_path: str = "fcnet_model.pth",
                 verbose: bool = False):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.__name__ = "FCNet"
        self.device = device

    def create_model(self, input_size):
        self.model = FCNet(input_size, self.num_classes, self.dropout_rate).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

        print("\nNew model created with input size:", input_size)

        if self.verbose:
            summary(self.model, (input_size,))
            print()

    def load_model(self):
        if os.path.exists(self.model_path):
            if self.model is None:
                raise ValueError("Model structure must be created before loading weights.")

            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print(f"Model loaded successfully from \'{self.model_path}\'.")
        else:
            print("No saved model found.")

    def train_model(self, train_loader, epochs=4000, learning_rate=0.0005):
        if self.model is None:
            raise ValueError("Model must be created before training.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if self.verbose:
                if epoch % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}], Training Loss: {running_loss / len(train_loader)}")

    def test_model(self, test_loader):
        if self.model is None:
            raise ValueError("Model must be created before testing.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return predictions

    def save_model(self, model_path: str):
        if self.model is None:
            raise ValueError("Model must be created before saving.")

        torch.save(self.model.state_dict(), model_path)
        print(f">> Model saved to \'{model_path}\'")
