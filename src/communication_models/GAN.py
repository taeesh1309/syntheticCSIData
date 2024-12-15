import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# File paths
data_path = "src/communication_models/data"
csv_files = [
    os.path.join(data_path, f"combined_csi_results_part_{i}.csv") for i in range(1, 193)
]

# Custom dataset
class CSIDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.encoder = OneHotEncoder(sparse_output=False)  # Initialize OneHotEncoder
        self.scaler = MinMaxScaler()  # Initialize MinMaxScaler
        self.data, self.targets = self.load_and_preprocess()

    def load_and_preprocess(self):
        # Combine multiple CSVs into one DataFrame
        dataframes = [pd.read_csv(file) for file in self.file_paths]
        df = pd.concat(dataframes, ignore_index=True)

        # Separate features and target (I and Q values)
        X = df.drop(columns=["real", "imag", "SNR"]) 
        Y = df[["real", "imag"]]  # Target (CSI values)

        # Identify categorical and numerical columns
        categorical_columns = ["antenna_geometry", "scenario"]  # Adjust based on dataset
        numerical_columns = X.columns.difference(categorical_columns)

        # One-hot encode categorical features
        categorical_data = self.encoder.fit_transform(X[categorical_columns])

        # Normalize numerical features
        numerical_data = self.scaler.fit_transform(X[numerical_columns])

        # Combine processed categorical and numerical features
        processed_X = np.hstack((categorical_data, numerical_data))

        return torch.tensor(processed_X, dtype=torch.float32), torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Initialize dataset and dataloader
batch_size = 64
dataset = CSIDataset(csv_files)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = dataset[0][0].shape[0]  # Feature size after preprocessing
output_dim = 2  # CSI (I and Q values)
epochs = 500
lr = 0.0002

# Initialize models and optimizers
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(input_dim + output_dim)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
adversarial_loss = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        batch_size = data.size(0)

        # Real data
        real_data = torch.cat((data, target), dim=1)
        real_labels = torch.ones((batch_size, 1))

        # Fake data
        fake_target = generator(data)
        fake_data = torch.cat((data, fake_target), dim=1)
        fake_labels = torch.zeros((batch_size, 1))

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_data), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(fake_data), real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save the trained generator model
torch.save(generator.state_dict(), "generator.pth")

# Testing the generator
generator.eval()
with torch.no_grad():
    test_features = torch.tensor([...], dtype=torch.float32)  # Replace with test input
    predicted_csi = generator(test_features)
    print("Predicted CSI (I, Q):", predicted_csi)
