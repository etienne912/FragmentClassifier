import os
import sys
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg


def file_to_fragment(path,return_path):
    path = path
    file_path = []
    file_class = []

    for subdirectory in os.walk(path):
        for file in subdirectory[2]:
            file_path.append(os.path.join(subdirectory[0], file))
            file_class.append(file.split(".")[1])
    df = pd.DataFrame({"class": file_class, "path": file_path})
    path_sample = return_path

    for cls in ["gz"]:
        path_folder = os.path.join(path_sample + "/" + cls)
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)
        count = 0
        for file_path in df[df["class"] == cls][["path"]].values:
            file = open(file_path[0], "rb")

            filesize = os.stat(file_path[0]).st_size
            nb_images = (filesize - 1024) // 4096
            if nb_images != 0:
                file.read(512)
                data = file.read(4096)

                for i in range(nb_images):
                    flatNumpyArray = np.array(bytearray(data))
                    grayImage = flatNumpyArray.reshape(64, 64)
                    backtorgb = cv2.cvtColor(grayImage,cv2.COLOR_GRAY2RGB)
                    cv2.imwrite(os.path.join(path_folder, str(count) + '.jpg'), backtorgb)
                    data = file.read(4096)

                    count += 1

def read_fragment_to_dataframe(path:str):
    cls = []
    image = []
    for subdirectory in os.walk(path):
        for image in subdirectory[2]:
            cls.append(subdirectory)
            img = mpimg.imread(os.path.join(subdirectory[0], image))
            print(img)

if __name__ == '__main__':
    path_directory ="../data/fragment"
    path = "../data/govdocs_selected"
    file_to_fragment(path=path,return_path=path_directory)


