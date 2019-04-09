import os

files = os.listdir("Bboxes_mydataset")

for file in files:
	path = os.path.join("Bboxes_mydataset", file)
	name = file[7:-4]
	index = name.zfill(8)
	movie = file[:1]
	movie_index = movie.zfill(2)
	new_name = movie_index + "_Frame_" + index + file[-4:]
	new_path = os.path.join("Bboxes_mydataset", new_name)
	os.rename(path, new_path)