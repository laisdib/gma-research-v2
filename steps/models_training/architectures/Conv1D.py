import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary

from steps.models_training.architectures.FCNet import FCNet


class Conv1DFCNet(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(Conv1DFCNet, self).__init__()

        # Layers with old settings
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=3)
        # self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layers with old settings
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3)
        # self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()

        conv_output_size = self._get_conv_output_size(input_size)
        self.fcnet = FCNet(input_size=conv_output_size, num_classes=num_classes, dropout_rate=dropout_rate)

    def _get_conv_output_size(self, input_size):
        """
        Calculate output size of convolutional layers.
        """

        with torch.no_grad():
            x = torch.zeros(1, 1, input_size)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            output_size = x.shape[1]

        return output_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)

        x = self.fcnet(x)
        return x


class Conv1DFCNetModel:
    def __init__(self, num_classes: int, device, dropout_rate: float = 0.5, model_path: str = "conv1dfcnet_model.pth",
                 verbose: bool = False):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.__name__ = "Conv1DFCNet"
        self.device = device

    def create_model(self, input_size):
        self.model = Conv1DFCNet(input_size, self.num_classes, self.dropout_rate).to(self.device)
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
        print(f'Model saved to \'{model_path}\'')
