#import cv2
import os
import matplotlib.pyplot as plt
import csv
from natsort import natsorted

# paths = natsorted(os.listdir('predictions_train'))

# for path in paths:
# 	if int(path[0]) >= 5:
# 		file = os.path.join('predictions_train', path)
# 		img = plt.imread(file)
# 		plt.imshow(img)
# 		plt.title(path)
# 		plt.show()

# csv_paths = natsorted(os.listdir('Bboxes_mydataset'))

# for file in csv_paths:
# 	if '9_Frame' in file: # and int(file[7:-4])>=1320 and int(file[7:-4])<=1800:
# 		print(file)
# 		file = os.path.join('Bboxes_mydataset', file)
# 		with open(file, "r") as f:
# 		    data = list(csv.reader(f))


	    	
# 		with open(file, "w") as f:
# 			for row in data:
# 				#if float(row[2]) > 1200:
# 				row[1] = '2.0'
# 				if abs(float(row[2]) - float(row[4])) < 1000:
# 					writer = csv.writer(f)
# 					writer.writerow(row)


