import multiprocessing
import cv2
import os
import scipy.misc 
from PIL import Image
import numpy as np


NEW_WIDTH = 512
NEW_HEIGHT = 512#target shape

FOLDER = '/server_space/jiangyl/zhu_useful/help/'#the pics in the folder you want to divide
SAVE_FOLDER = '/server_space/jiangyl/zhu_useful/help/img/'# save path
TARGET_TYPE = '.png'#save file format
def divide_photo(photos):
	if photos[-3:] == 'peg':#the format of the pics you want to divide
		count = 0
		sets = []
		print(os.path.join(FOLDER,photos))
		img = cv2.imread(os.path.join(FOLDER,photos),1)#format!
		height = img.shape[0]
		width = img.shape[1]
		col = 0
		while 1:
			row = 0
			if height - col > NEW_HEIGHT:
				while 1:
					if width - row > NEW_WIDTH:
						col_b = col + NEW_HEIGHT
						row_b = row + NEW_WIDTH
						sets.append(img[col:col_b, row:row_b])
					else:
						col_b = col + NEW_HEIGHT
						row_b = width
						row = row_b - NEW_WIDTH
						sets.append(img[col:col_b, row:row_b])
						break
					row += NEW_WIDTH
			else:
				while 1:
					
					col_b = height
					col = col_b - NEW_HEIGHT
					if width - row > NEW_WIDTH:
						row_b = row + NEW_HEIGHT
						sets.append(img[col:col_b, row:row_b])
					else:
						row_b = width
						row = row_b - NEW_WIDTH
						sets.append(img[col:col_b, row:row_b])
						break	
					row = row + NEW_WIDTH
				break
			col += NEW_HEIGHT
	for each in sets:
		filename = photos[:-5]
		filename = filename+'RGB'#name of saved pics
		filename = filename+'-'+str(count)
		f_name = SAVE_FOLDER+filename.lower()+TARGET_TYPE
		b, g, r = cv2.split(each)
		each = cv2.merge([r, g ,b])
		scipy.misc.imsave(f_name,each)
		print(filename)
		count += 1

if __name__ == '__main__':
	multiprocessing.freeze_support()
	photos = os.listdir(FOLDER)
	p = multiprocessing.Pool(10)#multi processes enable
	for each in photos:
		p.apply_async(divide_photo, args=(each,))
	p.close()
	p.join()
