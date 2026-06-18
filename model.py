
import torch
import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # CHANGE: The input_size must be 126 to match your hand landmark data
        self.lstm = nn.LSTM(126, 256, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length, 126)
        out, (hn, _) = self.lstm(x)
        # We take the output of the last hidden state for classification
        return self.fc(hn[-1])
