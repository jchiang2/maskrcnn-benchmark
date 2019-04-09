import cv2
import os
import csv
import numpy as np
from natsort import natsorted
from shutil import copyfile
from collections import defaultdict

def overwrite_csv():

	csv_paths = natsorted(os.listdir('Bboxes_mydataset'))

	for file in csv_paths:
		if '10_Frame' in file and int(file[8:-4])>=12800 and int(file[8:-4])<=13040:
			print(file)
			file = os.path.join('Bboxes_mydataset', file)
			with open(file, "r") as f:
			    data = list(csv.reader(f))

			with open(file, "w") as f:
				for row in data:
					if int(row[0]) == 0:
						row[1] = '2.0'
					if abs(float(row[2]) - float(row[4])) < 1000:
						writer = csv.writer(f)
						writer.writerow(row)

def split():
	imgPath = natsorted(os.listdir('Frames'))
	csvPath = natsorted(os.listdir('Bboxes_mydataset'))
	npyPath = natsorted(os.listdir('opt_flow_np'))

	numFiles = len(imgPath)
	randInd = np.random.randint(numFiles, size=int(numFiles/2))

	splitImg = [imgPath[i] for i in randInd]
	splitCsv = [csvPath[i] for i in randInd]
	splitNpy = [npyPath[i] for i in randInd]

	for (img, label, flow) in zip(splitImg, splitCsv, splitNpy):
		copyfile("Frames/{}".format(img),
				  "split_train/Frames/{}".format(img))
		copyfile("Bboxes_mydataset/{}".format(label),
				  "split_train/Bboxes_mydataset/{}".format(label))
		copyfile("opt_flow_np/{}".format(flow),
				  "split_train/opt_flow_np/{}".format(flow))

		print(img, label, flow)

def remove_dup():
	splitImg = natsorted(os.listdir('split_train/Frames'))
	imgPath = natsorted(os.listdir('Frames'))
	csvPath = natsorted(os.listdir('Bboxes_mydataset'))
	npyPath = natsorted(os.listdir('opt_flow_np'))

	imgDict = defaultdict(int)

	for file in splitImg:
	    imgDict[file[:-4]] = 1
	# Remove image files in csv dict
	for file in imgPath:
	    if imgDict[file[:-4]] == 1:
	        file = os.path.join('Frames',file)
	        os.remove(file)
	        print("Deleted image {}".format(file))
	for file in csvPath:
	    if imgDict[file[:-4]] == 1:
	        file = os.path.join('Bboxes_mydataset',file)
	        os.remove(file)
	        print("Deleted label {}".format(file))
	# Remove optFlow files in csv dict
	for file in npyPath:
	    if imgDict[file[:-4]] == 1:
	        file = os.path.join('opt_flow_np',file)
	        os.remove(file)
	        print("Deleted optFlow {}".format(file))

def find_err():
	csvPath = natsorted(os.listdir('split_train/Bboxes_mydataset'))
	for file in csvPath:
		file = os.path.join('split_train/Bboxes_mydataset',file)
		infoArray = np.genfromtxt(file, delimiter = ',')
		infoArray = np.reshape(infoArray, (-1,6))
		if infoArray.shape[1] > 2:
			boxes = infoArray[:,2:]
		labels = infoArray[:,1]
		labels[labels==3.] = 1.0
		for i in labels:
			if i != 1.0 and i != 2.0:
				print(file)
				break

def main():
	find_err()

if __name__ == "__main__":
    main()


