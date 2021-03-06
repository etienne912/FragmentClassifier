import math
import os
import sys

import cv2
import numpy
import numpy as np
import torch
from torchvision.transforms import transforms

from network import Net

files_types = ['txt', 'ppt', 'pdf', 'doc', 'jpg', 'gz', 'html', 'ps', 'xls']
fragments_types = dict(
	{'txt': 0, 'ppt': 0, 'pdf': 0, 'doc': 0, 'jpg': 0, 'gz': 0, 'html': 0, 'ps': 0, 'xls': 0, 'unk': 0})
fragments_types_colors = dict(
	{'txt': (255, 255, 255), 'ppt': (128, 128, 128), 'pdf': (0, 0, 255), 'doc': (255, 0, 0),
	 'jpg': (0, 255, 255), 'gz': (0, 0, 128), 'html': (255, 255, 0), 'ps': (128, 0, 0), 'xls': (0, 255, 0),
	 'unk': (255, 0, 255)})

# txt: White
# ppt: Gray
# pdf: Red
# doc: Blue
# jpg: Yellow
# gz: Maroon
# html: Cyan
# ps: Navy
# xls: Lime
# unk: Magenta


if __name__ == '__main__':
	input_path = sys.argv[1]
	output_name = sys.argv[2]
	try:

		device = ("cuda" if torch.cuda.is_available() else "cpu")

		with torch.no_grad():
			model = Net()
			model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
			model = model.to(device)
			model.eval()

			file = open(input_path, "rb")

			filesize = os.stat(input_path).st_size
			nb_images = filesize // 4096
			if nb_images == 0:
				exit()

			prediction_list = []
			list_fragments_types = []
			list_color_fragments_types = []
			data = file.read(4096)
			for i in range(nb_images):
				flatNumpyArray = numpy.array(bytearray(data))
				if flatNumpyArray.argmax() == 0:
					prediction_list.append(None)
					data = file.read(4096)
					continue
				grayImage = flatNumpyArray.reshape(64, 64)
				transform = transforms.Compose(
					[transforms.ToTensor()])

				img = transform(grayImage)
				x = torch.unsqueeze(img, 0)

				prediction_list.append(model(x))
				data = file.read(4096)

			for i in range(len(prediction_list)):
				current = prediction_list[i]
				if current is None:
					list_color_fragments_types.append((0, 0, 0))
					continue

				predicted_class = np.argmax(current)
				if current[0][predicted_class] < 0.8:
					previous = prediction_list[i] if i > 1 else [0] * len(files_types)
					next = prediction_list[i] if i < len(prediction_list) else [0] * len(files_types)
					previous[0] = previous[0] * 0.75
					next[0] = next[0] * 0.75

					proceed = np.add(previous, np.add(current, next))
					predicted_class = np.argmax(proceed)

					if proceed[0][predicted_class] > 1.2:
						fragments_types[files_types[predicted_class]] += 1
						list_fragments_types.append(files_types[predicted_class])
						list_color_fragments_types.append(fragments_types_colors[files_types[predicted_class]])
					else:
						fragments_types["unk"] += 1
						list_color_fragments_types.append(fragments_types_colors['unk'])
						list_fragments_types.append("unk")
				else:
					fragments_types[files_types[predicted_class]] += 1
					list_fragments_types.append(files_types[predicted_class])
					list_color_fragments_types.append(fragments_types_colors[files_types[predicted_class]])

			if not os.path.exists('output'):
				os.makedirs('output')

			with open('output/' + output_name + '.txt', 'w+') as f:
				f.write("File type | Number of fragments | Total size (KB)\n")
				f.write("----------|---------------------|----------------\n")
				for item, value in fragments_types.items():
					f.write("%s | %s | %s\n" % (item, value, math.floor(value * 4.096)))

			size = math.ceil(math.sqrt(nb_images))
			for i in range((size * size) - nb_images):
				list_color_fragments_types.append((0, 0, 0))

			flatNumpyArray = numpy.array(list_color_fragments_types)
			image = flatNumpyArray.reshape((size, size, 3))
			cv2.imwrite('output/' + output_name + '.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

	finally:
		file.close()
