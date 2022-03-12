"""
https://ensemble-pytorch.readthedocs.io/en/latest/quick_start.html
"""

import torch.nn as nn
from torch.nn import functional as F
from torchensemble.utils.logging import set_logger
from torchensemble import VotingClassifier
from torchensemble.utils import io


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, data):
        data = data.view(data.size(0), -1)  # flatten
        output = F.relu(self.linear1(data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output


logger = set_logger('classification_mnist_mlp')

model = VotingClassifier(
    estimator=MLP,
    n_estimators=10,
    cuda=True,
)

criterion = nn.CrossEntropyLoss()
model.set_criterion(criterion)

model.set_optimizer('Adam',             # parameter optimizer
                    lr=1e-3,            # learning rate of the optimizer
                    weight_decay=5e-4)  # weight decay of the optimizer

# Training
model.fit(train_loader=train_loader,  # training data
          epochs=100)                 # the number of training epochs

# Evaluating
accuracy = model.predict(test_loader)

io.load(new_ensemble, save_dir)  # reload
