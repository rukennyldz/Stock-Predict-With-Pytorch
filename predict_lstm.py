import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel

# Load the stock data from CSV
df = pd.read_csv("data/apple_stock.csv")

# Parse date column
df['Date'] = pd.to_datetime(df['Date'])  # <-- Tarih formatını tanır
df.set_index('Date', inplace=True)       # <-- Tarihi indeks yapar

data = df[['Close']].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create input sequences and labels
sequence_length = 60
X, y = [], []
dates = []  # Tarihleri burada toplayacağız

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i])
    y.append(scaled_data[i])
    dates.append(df.index[i])  # Date'leri burada topluyoruz

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
model.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))
model.eval()

# Make predictions
predictions = []
with torch.no_grad():
    for i in range(len(X)):
        input_seq = X[i].unsqueeze(0).to(device)
        output = model(input_seq)
        predictions.append(output.cpu().numpy())

predictions = np.array(predictions).reshape(-1, 1)

# Inverse transform to get original price values
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y.numpy())

# Plot the results with proper dates
plt.figure(figsize=(12, 6))
plt.plot(dates, actual_prices, label="Actual Price", color='blue')
plt.plot(dates, predicted_prices, label="Predicted Price", color='orange')
plt.title("Stock Price Prediction vs Actual Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/full_prediction_plot.png")
plt.show()

print("📊 Full prediction plot saved as: plots/full_prediction_plot.png")
