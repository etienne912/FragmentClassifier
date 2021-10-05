import os
import sys
import cv2

import numpy

if __name__ == '__main__':
	global file
	try:
		path = sys.argv[1]

		file = open(path, "rb")

		filesize = os.stat(path).st_size
		nb_images = (filesize - 1000) // 4096
		if nb_images == 0:
			exit()

		file.read(512)
		data = file.read(4096)
		for i in range(nb_images):
			flatNumpyArray = numpy.array(bytearray(data))
			grayImage = flatNumpyArray.reshape(64, 64)
			cv2.imwrite(path + str(i) + '.jpg', grayImage)

			data = file.read(4096)

	finally:
		file.close()
