import torch
import torch.nn as nn
from torch.nn import Linear


class NN_Model_Template(nn.Module):
    def __init__(self):
        super(NN_Model_Template, self).__init__()

        self.device = None
        self.test_label_output = None

    def out(self, data):
        assert False, "Please implement the out function"

    def predict(self, test_features):
        with torch.no_grad():
            test_features = torch.from_numpy(test_features).to(self.device)
            outputs = self.out(test_features)

            self.test_label_output = outputs.cpu().numpy().round().flatten()

        return self.test_label_output

    # initialize the parameters of the network
    def init_weights(self, m):
        assert False, "Please implement the init_weights function"

    def score(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).mean()

    def correct_num(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).sum()


class QuadraticNN_Model(NN_Model_Template):
    def __init__(self, n_features=1):
        super(QuadraticNN_Model, self).__init__()

        h_size = 256

        self.W = nn.Linear(n_features, h_size, bias=False)
        self.v = nn.Linear(h_size, 1, bias=True)

        self.m = torch.nn.Softmax(dim=1)

        # Linear pathway
        self.linear = nn.Linear(n_features, 1, bias=True)

    def out(self, input):
        # proceed the input layer
        z = self.W(input)
        z2 = z ** 2
        quadratic_output = self.v(z2)

        linear_output = self.linear(input)

        x = quadratic_output.squeeze(-1) + linear_output.squeeze(-1)

        return x

    # initialize the parameters of the network
    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)

            if m.bias is not None:
                m.bias.data.fill_(0.02)
    
    def predict(self, test_features):
        with torch.no_grad():
            test_features = torch.from_numpy(test_features).to(self.device)
            outputs = self.out(test_features)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            self.test_label_output = preds.cpu().numpy().round().flatten()

        return self.test_label_output




                