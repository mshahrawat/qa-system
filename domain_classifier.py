import torch.nn as nn

# 1. Input to Hidden Layer : Linear(input dim, hidden dim)
# 2. Tanh activation, ReLU
# 3. Hidden to Output Layer : Linear(hidden dim, 2)
# 4. LogSoftmax on top of the output layer

class DomainClassif(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DomainClassif, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        hidden = self.linear(x)
        hidden = self.relu(hidden)
        hidden = self.hidden(hidden)
        output = self.softmax(hidden)
        return output