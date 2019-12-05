import sys
import os
import pandas as pd

DATA_DIR = './jaffe'

files = []
for f in os.listdir(DATA_DIR):
    if f.split(".")[-1] == "tiff":
        components = [f] + f.split(".")[:-1]
        files.append(components)

df = pd.DataFrame(files, columns = ["filename", "component1", "component2", "index"])
df['key'] = df['component1'] + '-' + df['component2']

labels = open("jaffe/labels.txt", "r")
array = []
for line in labels:
    if line == "": break
    components = line.split(" ")
    key = components[-1][:-1]
    labels = components[:-1]
    for lab in labels:
        lab = float(lab)
    array.append(labels + [key])

#print(files)
# HAP SAD SUR ANG DIS FEA
labels = pd.DataFrame(array, columns = ['index', "HAP", "SAD", "SUR", "ANG", "DIS", "FEA", "key"])
result = df.join(labels, lsuffix='_left')
print(result)
result.to_csv(os.path.join(DATA_DIR, "labels.csv"))
