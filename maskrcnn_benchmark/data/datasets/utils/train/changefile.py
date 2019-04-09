import os

folder = "train/train/Frames"

files = os.listdir(folder)

for file in files:
	assert len(file) < 20
	path = os.path.join(folder, file)
	name = file[7:-4]
	index = name.zfill(8)
	movie = file[:1]
	movie_index = movie.zfill(2)
	new_name = movie_index + "_Frame_" + index + file[-4:]
	new_path = os.path.join(folder, new_name)
	# os.rename(path, new_path)
