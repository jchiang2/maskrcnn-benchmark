import os
import cv2 as cv
import csv
import argparse

# fn = '/home/jchiang2/github/data_maskrcnn/000002-video.mp4'

# cam = cv.VideoCapture(fn)
# frame_cnt = 0


# while True:

#     ret, img = cam.read()
#     if frame_cnt % 20 == 0:
#     	cv.imwrite("dataApr2/12_Frame_{}.jpg".format(str(frame_cnt).zfill(8)), img)
#     	print("Saving {}".format(frame_cnt))
#     if not ret:
#     	break
#     frame_cnt += 1
parser = argparse.ArgumentParser(description="Visualize Predictions")
parser.add_argument("--root", default="train/Bboxes", type=str)
args = parser.parse_args()

dir = args.root
files = sorted(os.listdir("{}".format(dir)))

for file in files:
	if int(file[-9:-4]) >= 0:
		file = os.path.join(dir, file)
		print(file)

		# rows = []
		# with open(file, 'r', newline='') as readfile:
		# 	reader = csv.reader(readfile, delimiter=',')
		# 	for row in reader:
		# 		rows.append(row)


		# with open(file, 'w', newline='') as writefile:
		# 	writer = csv.writer(writefile, delimiter=',')
		# 	for row in rows:
		# 		print(row)
		# 		if row[1] == '3.0':
		# 			row[1] = '1.0'
		# 		writer.writerow(row)



    # infoArray = np.genfromtxt(path, delimiter = ',')

    # infoArray = np.reshape(infoArray, (-1,6))
    # if infoArray.shape[1] > 2:
    #     boxes = infoArray[:,2:]

    # # Labels should be torch.long() type
    # labels = infoArray[:,1]
    # labels[labels==3.] = 1.0
    # labels = torch.from_numpy(labels).long()