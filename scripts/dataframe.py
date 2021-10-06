import os
import sys

if __name__ == '__main__':
    path = "../data/govdocs_selected"
    ret = []
    for subdirectory in os.walk(path):
        for file in subdirectory[2]:
            ret.append(os.path.join(subdirectory[0], file))


