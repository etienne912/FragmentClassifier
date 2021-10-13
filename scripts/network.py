import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
		self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(64, 126, 3, stride=1, padding=1)
		self.conv5 = nn.Conv2d(126, 256, 3, stride=1, padding=1)
		self.fc1 = nn.Linear(2304, 1500)
		self.fc2 = nn.Linear(1500, 500)
		self.fc3 = nn.Linear(500, 9)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = self.pool(F.relu(self.conv4(x)))
		x = self.pool(F.relu(self.conv5(x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = F.softmax(x, dim=1)
		return x
