import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import LSTMModel
import numpy as np
from datetime import timedelta

# Load the trained model
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

# Load and preprocess the data
df = pd.read_csv('data/apple_stock.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df['Close'].values.reshape(-1, 1)
data = (data - np.mean(data)) / np.std(data)  # normalize

window_size = 10
X = []
for i in range(len(data) - window_size):
    X.append(data[i:i+window_size])
X = torch.tensor(X, dtype=torch.float32)

# Predict the next value
with torch.no_grad():
    prediction = model(X[-1].unsqueeze(0)).item()

# Unnormalize prediction
prediction = prediction * np.std(df['Close'].values) + np.mean(df['Close'].values)
print(f"📈 Predicted next closing price: {prediction:.2f} USD")

# Add predicted value to plot
dates = df.index
next_day = dates[-1] + timedelta(days=1)

# Create a new DataFrame for plotting
predicted_series = pd.Series(df['Close'].values, index=dates)
predicted_series[next_day] = prediction

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(predicted_series.index, predicted_series.values, label='Actual + Predicted', color='blue')
plt.axvline(x=next_day, color='red', linestyle='--', label='Predicted Point')
plt.title('📊 Closing Price Prediction')
plt.xlabel('Date')  # <-- X axis label is now "Date"
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Save and show
plt.tight_layout()
plt.savefig('plots/latest_prediction_plot.png')
plt.show()
