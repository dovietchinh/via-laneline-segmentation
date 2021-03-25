import cv2
import numpy as np
import os
import pathlib


def transform_data(data_folder = './data/', phase = 'train'):
	global count
	try:
		assert phase in ['train','val'], "phase keyword must be 'train' or 'val'"
	except AssertionError as msg: 
		print(msg)
		
	old_mask_folder = data_folder + phase+'/masks/'
	old_image_folder = data_folder + phase +'/images/'
	new_image_folder = data_folder + phase+'/new_images/'
	new_mask_folder = data_folder + phase+ '/new_masks/'
	pathlib.Path(new_image_folder).mkdir(exist_ok=True, parents=True)
	pathlib.Path(new_mask_folder).mkdir(exist_ok=True, parents=True)
	image_names = os.listdir(old_image_folder)
	mask_names = os.listdir(old_mask_folder)
	
	for name in image_names:
		img = cv2.imread(old_image_folder+name,cv2.IMREAD_COLOR)
		img = cv2.resize(img,(256,256))
		test = cv2.imwrite(new_image_folder + name, img) 
		if test ==False:
			print( ' Error occur with image: ',name)
		else:
 			count +=1
 			print( 'processed images :',count,end='\r')
			
		

	for name in mask_names:
		img = cv2.imread(old_mask_folder+name,cv2.IMREAD_GRAYSCALE)
		thresh, img = cv2.threshold(img, thresh=140, maxval=255, 
			type=cv2.THRESH_BINARY)
		img = cv2.resize(img,(256,256))
		test = cv2.imwrite(new_mask_folder + name, img) 
		if test ==False:
			print( ' Error occur with image: ',name)
		else:
 			count +=1
 			print( 'processed images :',count,end='\r')	
if __name__ =='__main__':
	print('processing with train folder')
	count = 0
	data_folder = './data/'
	transform_data(data_folder , phase = 'train')
	print('total images in train_folder :',count)
	
	print('processing with val folder')
	count = 0
	transform_data(data_folder , phase = 'val')
	print('total images in val_folder :',count)
	print('Done!')
	
