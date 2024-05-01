import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

# Set page title
st.set_page_config(page_title="Tesla Stock Prediction")

# Add a title and description
st.title("Tesla Stock Prediction using Transformer Model")
st.write("This app predicts Tesla stock prices using a PyTorch Transformer model.")

@st.cache_data
def load_data():
    tesla_data = yf.download('TSLA', start='2018-01-01', end='2023-01-01')
    scaler = MinMaxScaler(feature_range=(0, 1))
    tesla_data['Normalized_Close'] = scaler.fit_transform(tesla_data['Close'].values.reshape(-1, 1))
    return tesla_data, scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.view(x.size(0), -1, self.hidden_size)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

@st.cache_resource
def train_model(train_X, train_y, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate):
    train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(2)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    model = TransformerModel(input_size=seq_length, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a placeholder in the sidebar for live updating
    progress_text = st.sidebar.empty()
    progress_epoch = st.sidebar.progress(0)

    for epoch in range(num_epochs):
        permutation = torch.randperm(train_X.size()[0])
        for i in range(0, train_X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = train_X[indices], train_y[indices]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            progress_epoch.progress((i + 1) / len(train_X))
            # Update the text in the sidebar with the latest progress
            if i % 1024 == 0:
                progress_text.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_X)}], Loss: {loss.item():.4f}')

    return model

# Load and prepare the data
tesla_data, scaler = load_data()
train_size = int(len(tesla_data) * 0.8)
train_data = tesla_data[:train_size]
test_data = tesla_data[train_size:]

# Ensure the input data is properly formatted
seq_length = 20

# Create sequences for training and testing sets
train_X, train_y = create_sequences(train_data['Normalized_Close'].values, seq_length)
test_X, test_y = create_sequences(test_data['Normalized_Close'].values, seq_length)

# Get user input for training parameters
st.sidebar.header("Training Parameters")
hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=1024, value=512, step=32)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=10, value=3, step=1)
num_heads = st.sidebar.slider("Number of Heads", min_value=1, max_value=8, value=4, step=1)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.1)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=500, value=1, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")

# Train the model (cached)
model = train_model(train_X, train_y, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)

if st.button("Rerun Training"):
    model = train_model(train_X, train_y, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)
    st.experimental_rerun()

# Evaluate the model on the testing set
test_X_tensor = torch.tensor(test_X, dtype=torch.float32).unsqueeze(2)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(test_X_tensor)
    test_loss = nn.MSELoss()(test_outputs.squeeze(), test_y_tensor)
    st.write(f'Test Loss: {test_loss.item():.4f}')

# Make predictions on the testing set
with torch.no_grad():
    test_predicted = model(test_X_tensor)

# Visualize the results on the testing set
test_actual_data = pd.DataFrame(test_y, columns=['Actual'])
test_predicted_data = pd.DataFrame(test_predicted.numpy(), columns=['Predicted'])
test_chart_data = pd.concat([test_actual_data, test_predicted_data], axis=1)
st.line_chart(test_chart_data)

# Add a button to download the model
if st.button("Download Model"):
    model_path = "transformer_model.pt"
    torch.save(model.state_dict(), model_path)
    with open(model_path, "rb") as f:
        bytes = f.read()
        st.download_button(
            label="Download Model",
            data=bytes,
            file_name=model_path,
            mime="application/octet-stream",
        )
    os.remove(model_path)
