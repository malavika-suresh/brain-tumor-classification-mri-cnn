import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils
import matplotlib.pyplot as plt
from os import listdir
import time


def hms_string(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return f"{h}:{m}:{round(s,1)}"

def augment_data(file_dir, n_generated_samples, save_to_dir):


	data_gen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,brightness_range=(0.3,1.0),horizontal_flip=True,vertical_flip=True,fill_mode='nearest')


	for filename in listdir(file_dir):

		image = cv2.imread(file_dir + '/' + filename)

		image = image.reshape((1,)+image.shape)
		save_prefix = 'aug_' + filename[:-4]

		i=0
		for batch in data_gen.flow(x=image, batch_size=1,save_to_dir=save_to_dir,save_prefix=save_prefix,save_format='jpg'):
			i += 1
			if i > n_generated_samples:
				break

def data_summary(main_path):

	yes_path = main_path+'yes'
	no_path = main_path+'no'


	m_pos = len(listdir(yes_path))
	m_neg = len(listdir(no_path))

	m = (m_pos+m_neg)

	pos_prec = (m_pos* 100.0)/ m
	neg_prec = (m_neg* 100.0)/ m

	print(f"Number of examples: {m}")
	print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {m_pos}") 
	print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {m_neg}")
	return yes_path, no_path

start_time = time.time()
orig_data= 'brain_tumor_dataset/'
augmented_data_path = 'augmented_data/'
yes_path, no_path = data_summary(orig_data)

print(yes_path)

augment_data(file_dir=yes_path, n_generated_samples=6, save_to_dir=augmented_data_path+'yes')
augment_data(file_dir=no_path, n_generated_samples=9, save_to_dir=augmented_data_path+'no')

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")



