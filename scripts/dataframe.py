import os
import sys
from pyspark.sql import SparkSession
import cv2
import numpy as np
import pandas as pd
import os



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

    for cls in ["csv", "pdf", "txt", "html", "doc", "ppt", "xls"]:
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
                    cv2.imwrite(os.path.join(path_folder, str(count) + '.jpg'), grayImage)
                    data = file.read(4096)

                    count += 1

if __name__ == '__main__':
    spark = SparkSession.builder.appName("DL with Spark Deep Cognition").getOrCreate()
    sc = spark.sparkContext
    image_df = spark.read.format("image").load("../data/fragment/txt")





