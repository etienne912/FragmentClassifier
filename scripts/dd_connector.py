import os
import sys

import cv2
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import math

files_types = ['txt', 'ppt', 'pdf', 'doc', 'jpg', 'gz', 'html', 'ps', 'xls']
fragments_types = dict(
	{'txt': 0, 'ppt': 0, 'pdf': 0, 'doc': 0, 'jpg': 0, 'gz': 0, 'html': 0, 'ps': 0, 'xls': 0, 'unk': 0})
fragments_types_colors = dict(
	{'txt': (255, 255, 255), 'ppt': (128, 128, 128), 'pdf': (0, 0, 255), 'doc': (255, 0, 0),
	 'jpg': (255, 255, 0), 'gz': (0, 255, 0), 'html': (0, 255, 255), 'ps': (128, 0, 0), 'xls': (0, 0, 128),
	 'unk': (255, 0, 255)})

# txt: White
# ppt: Gray
# pdf: Red
# doc: Blue
# jpg: Yellow
# gz: Lime
# html: Cyan
# ps: Maroon
# xls: Navy
# unk: Magenta


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


if __name__ == '__main__':
	input_path = sys.argv[1]
	return_path = sys.argv[2]
	try:

		device = ("cuda" if torch.cuda.is_available() else "cpu")

		with torch.no_grad():
			model = Net()
			model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
			model = model.to(device)
			model.eval()
			print(model)

			file = open(input_path, "rb")

			filesize = os.stat(input_path).st_size
			nb_images = filesize // 4096
			if nb_images == 0:
				exit()

			list_fragments_types = []
			list_color_fragments_types = []
			data = file.read(4096)
			for i in range(nb_images):
				flatNumpyArray = numpy.array(bytearray(data))
				grayImage = flatNumpyArray.reshape(64, 64)
				transform = transforms.Compose(
					[transforms.ToTensor()])

				img = transform(grayImage)
				x = torch.unsqueeze(img, 0)

				prediction = model(x)
				predicted_class = np.argmax(prediction)
				if prediction[0][predicted_class] > 0.85:
					fragments_types[files_types[predicted_class]] += 1
					list_fragments_types.append(files_types[predicted_class])
					list_color_fragments_types.append(fragments_types_colors[files_types[predicted_class]])
				else:
					fragments_types["unk"] += 1
					list_color_fragments_types.append(fragments_types_colors['unk'])
					list_fragments_types.append("unk")

				data = file.read(4096)

			with open(return_path + '.txt', 'w+') as f:
				for item, value in fragments_types.items():
					f.write("%s | %s\n" % (item, value))

			size = math.ceil(math.sqrt(nb_images))
			for i in range((size * size) - nb_images):
				list_color_fragments_types.append((0, 0, 0))

			flatNumpyArray = numpy.array(list_color_fragments_types)
			grayImage = flatNumpyArray.reshape((size, size, 3))
			cv2.imwrite(return_path + '.png', grayImage, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

	finally:
		file.close()
