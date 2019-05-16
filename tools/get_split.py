from collections import defaultdict
import os
from shutil import copyfile

splitFiles = os.listdir("/home/jchiang2/github/data_maskrcnn/traintest_split/Frames")
splitDict = defaultdict(int)


for file in splitFiles:
	fileID = os.path.splitext(file)
	splitDict[fileID] = 1

videoFiles = os.listdir("/home/jchiang2/github/data_maskrcnn/train/OptFlow")
for file in videoFiles:
	fileID = os.path.splitext(file)
	if splitDict[fileID] == 1:
		print(file)
		origin = os.path.join("/home/jchiang2/github/data_maskrcnn/train/OptFlow", file)
		dest = os.path.join("/home/jchiang2/github/data_maskrcnn/traintest_split/OptFlow", file)
		print(origin)
		print(dest)
		# os.rename(origin, dest)
		copyfile(origin, dest)