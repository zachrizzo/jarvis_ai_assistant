import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Set page title
st.set_page_config(page_title="Tesla Stock Prediction")

# Add a title and description
st.title("Tesla Stock Prediction using Transformer Model")
st.write("This app predicts Tesla stock prices using a PyTorch Transformer model.")

# Load and prepare the data
tesla_data = yf.download('TSLA', start='2018-01-01', end='2023-01-01')
scaler = MinMaxScaler(feature_range=(0, 1))
tesla_data['Normalized_Close'] = scaler.fit_transform(tesla_data['Close'].values.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32).unsqueeze(2), torch.tensor(ys, dtype=torch.float32)

# Ensure the input data is properly formatted
seq_length = 20
X, y = create_sequences(tesla_data['Normalized_Close'].values, seq_length)

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

model = TransformerModel(input_size=seq_length, hidden_size=256, num_layers=10, num_heads=4, dropout=0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
batch_size = 32

# Create a placeholder in the sidebar for live updating
progress_text = st.sidebar.empty()
progress_num = st.sidebar.progress(0)
for epoch in range(num_epochs):
    permutation = torch.randperm(X.size()[0])
    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X[indices], y[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

        # Update the text in the sidebar with the latest progress
        if i % 1024 == 0:
            progress_text.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(X)}], Loss: {loss.item():.4f}')

    # Update the progress bar in the sidebar
    progress_num.progress((epoch + 1) / num_epochs)




# Make predictions
with torch.no_grad():
    predicted = model(X)

# Visualize the results
actual_data = pd.DataFrame(y.numpy(), columns=['Actual'])
predicted_data = pd.DataFrame(predicted.numpy(), columns=['Predicted'])
chart_data = pd.concat([actual_data, predicted_data], axis=1)

st.line_chart(chart_data)
