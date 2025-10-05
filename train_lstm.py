import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from model import LSTMModel  # import your LSTM model from model.py

# ========== 1. Load and preprocess data ==========
DATA_PATH = "data/apple_stock.csv"
TARGET_COLUMN = "Close"
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

df = pd.read_csv(DATA_PATH)
data = df[[TARGET_COLUMN]].values  # Only use the target column

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save the scaler
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ========== 2. Create Dataset ==========
class StockDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.X = []
        self.y = []
        for i in range(sequence_length, len(data)):
            self.X.append(data[i-sequence_length:i])
            self.y.append(data[i])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = StockDataset(scaled_data, SEQUENCE_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== 3. Setup model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

# ========== 4. Training loop ==========
print("🚀 Training started...\n")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# ========== 5. Save the trained model ==========
torch.save(model.state_dict(), "models/lstm_model.pth")
print("\n✅ Trained model saved to: models/lstm_model.pth")

# ========== 6. Plot training loss ==========
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(losses, color='blue')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/lstm_loss.png")
print("📊 Loss plot saved to: plots/lstm_loss.png")
