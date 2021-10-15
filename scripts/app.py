import io
import math
import os
import zipfile
from io import BytesIO

import cv2
import numpy
import numpy as np
import torch
from flask import Flask, request, send_file
from torchvision.transforms import transforms

from network import Net

app = Flask(__name__, static_folder='./')

device = ("cuda" if torch.cuda.is_available() else "cpu")

model = Net()
model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
model = model.to(device)
model.eval()

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

# Upload files
@app.route('/analyse', methods=['POST'])
def upload():
	smooth = 'smooth' in request.args

	chunk_size = 4096
	prediction_list = []

	while True:
		chunk = request.stream.read(chunk_size)
		if len(chunk) < chunk_size:
			break

		with torch.no_grad():
			flatNumpyArray = numpy.array(bytearray(chunk))
			if flatNumpyArray.argmax() == 0:
				prediction_list.append(None)
				continue
			grayImage = flatNumpyArray.reshape(64, 64)
			transform = transforms.Compose(
				[transforms.ToTensor()])

			img = transform(grayImage)
			x = torch.unsqueeze(img, 0)

			prediction_list.append(model(x))

	list_color_fragments_types = []

	for i in range(len(prediction_list)):
		current = prediction_list[i]
		if current is None:
			list_color_fragments_types.append((0, 0, 0))
			continue

		predicted_class = np.argmax(current)
		if smooth:
			if current[0][predicted_class] < 0.8:
				previous = prediction_list[i] if i > 1 else [0] * len(files_types)
				next = prediction_list[i] if i < len(prediction_list) else [0] * len(files_types)
				previous[0] = previous[0] * 0.75
				next[0] = next[0] * 0.75

				proceed = np.add(previous, np.add(current, next))
				predicted_class = np.argmax(proceed)

				if proceed[0][predicted_class] > 1.2:
					fragments_types[files_types[predicted_class]] += 1
					list_color_fragments_types.append(fragments_types_colors[files_types[predicted_class]])
				else:
					fragments_types["unk"] += 1
					list_color_fragments_types.append(fragments_types_colors['unk'])
			else:
				fragments_types[files_types[predicted_class]] += 1
				list_color_fragments_types.append(fragments_types_colors[files_types[predicted_class]])
		else:
			if current[0][predicted_class] > 0.8:
				fragments_types[files_types[predicted_class]] += 1
				list_color_fragments_types.append(fragments_types_colors[files_types[predicted_class]])
			else:
				fragments_types["unk"] += 1
				list_color_fragments_types.append(fragments_types_colors['unk'])

	if not os.path.exists('output'):
		os.makedirs('output')

	txt = "File type | Number of fragments | Total size (KB)\n"
	txt += "-------------------------------------------------\n"
	txt += str.join('', [("%s | %s | %s\n" % (item, value, math.floor(value * 4.096))) for item, value in fragments_types.items()])
	result = io.StringIO(txt)

	size = math.ceil(math.sqrt(len(list_color_fragments_types)))
	for i in range((size * size) - len(list_color_fragments_types)):
		list_color_fragments_types.append((0, 0, 0))

	flatNumpyArray = numpy.array(list_color_fragments_types)
	grayImage = flatNumpyArray.reshape((size, size, 3))

	is_success, buffer = cv2.imencode(".png", grayImage, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
	img = BytesIO(buffer)

	mem_zip = BytesIO()

	with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
		zf.writestr("result.txt", result.getvalue())
		zf.writestr("visu.png", img.getvalue())

	mem_zip.seek(0)

	return send_file(
		mem_zip,
		mimetype='application/x-zip',
		as_attachment=True,
		attachment_filename="result.zip"
	)


@app.route('/')
def root():
	return app.send_static_file('index.html')


if __name__ == "__main__":
	app.run()
