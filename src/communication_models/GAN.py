import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================== Dataset Class ==============================
class CSIDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = MinMaxScaler()
        self.data, self.targets = self.load_and_preprocess()

    def load_and_preprocess(self):
        # Combine multiple CSV files into one DataFrame
        dataframes = [pd.read_csv(file) for file in self.file_paths]
        df = pd.concat(dataframes, ignore_index=True)

        # Separate features and targets
        X = df.drop(columns=["real", "imag", "SNR"])
        Y = df[["real", "imag"]]

        # Identify categorical and numerical columns
        categorical_columns = ["antenna_geometry", "scenario"]
        numerical_columns = X.columns.difference(categorical_columns)

        # One-hot encode categorical features
        categorical_data = self.encoder.fit_transform(X[categorical_columns])

        # Normalize numerical features
        numerical_data = self.scaler.fit_transform(X[numerical_columns])

        # Combine processed features
        processed_X = np.hstack((categorical_data, numerical_data))

        return (
            torch.tensor(processed_X, dtype=torch.float32),
            torch.tensor(Y.values, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ============================== Model Classes ==============================
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ============================== Utility Functions ==============================
def preprocess_test_data(file_path, encoder, scaler):
    """
    Preprocess test dataset using the encoder and scaler from training.
    """
    df = pd.read_csv(file_path)

    # Separate features and targets
    X = df.drop(columns=["real", "imag", "SNR"])
    Y = df[["real", "imag"]]

    # Identify categorical and numerical columns
    categorical_columns = ["antenna_geometry", "scenario"]
    numerical_columns = X.columns.difference(categorical_columns)

    # One-hot encode categorical features
    categorical_data = encoder.transform(X[categorical_columns])

    # Normalize numerical features
    numerical_data = scaler.transform(X[numerical_columns])

    # Combine processed features
    processed_X = np.hstack((categorical_data, numerical_data))

    return (
        torch.tensor(processed_X, dtype=torch.float32),
        torch.tensor(Y.values, dtype=torch.float32),
    )


def save_model(generator, discriminator, optimizer_G, optimizer_D, path):
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
        },
        path,
    )
    print("Model saved to", path)


def load_model(generator, discriminator, optimizer_G, optimizer_D, path):
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    print("Model loaded from", path)


# ============================== Training Function ==============================
def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, epochs, loss_fn, model_path):
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
            real_loss = loss_fn(discriminator(real_data), real_labels)
            fake_loss = loss_fn(discriminator(fake_data.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = loss_fn(discriminator(fake_data), real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    save_model(generator, discriminator, optimizer_G, optimizer_D, model_path)


# ============================== Main Script ==============================
def main():
    # File paths
    data_path = "src/communication_models/data"
    test_file_path = "src/communication_models/testbed_data_updated.csv"
    model_path = "gan_model.pth"
    results_file_path = "test_results.csv"

    # Dataset and Dataloader
    csv_files = [
        os.path.join(data_path, f"combined_csi_results_part_{i}.csv") for i in range(1, 193)
    ]
    dataset = CSIDataset(csv_files)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model Initialization
    input_dim = dataset[0][0].shape[0]
    output_dim = 2
    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(input_dim + output_dim)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    loss_fn = nn.BCELoss()

    # Training or Loading the Model
    if os.path.exists(model_path):
        print("Loading the model...")
        load_model(generator, discriminator, optimizer_G, optimizer_D, model_path)
    else:
        print("Training the model...")
        train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, epochs=10, loss_fn=loss_fn, model_path=model_path)

    # Testing the Model
    generator.eval()
    with torch.no_grad():
        print("\nTesting the model...")
        test_data, test_targets = preprocess_test_data(
            test_file_path, dataset.encoder, dataset.scaler
        )
        predicted_csi = generator(test_data)

    # Evaluation
    mse = np.mean((predicted_csi.numpy() - test_targets.numpy()) ** 2)
    print(f"Mean Squared Error (MSE) on Test Data: {mse}")

    # Save results to a CSV file
    print("\nSaving test results to test_results.csv...")
    predicted_csi_np = predicted_csi.numpy()
    test_data_np = test_data.numpy()

    # Combine inputs and predicted CSI values
    results = np.hstack((test_data_np, predicted_csi_np))
    column_names = [f"input_{i}" for i in range(test_data_np.shape[1])] + ["predicted_I", "predicted_Q"]
    results_df = pd.DataFrame(results, columns=column_names)

    # Save to CSV
    results_df.to_csv(results_file_path, index=False)
    print(f"Test results saved to {results_file_path}")


if __name__ == "__main__":
    main()
