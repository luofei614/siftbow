#!/usr/local/bin/python2.7
#python findFeatures.py -t dataset/train/
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
#fig = plt.figure(figsize=(20,20))
#ax = plt.subplot(111,aspect = 'equal')
#plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

from sklearn import preprocessing
#from rootsift import RootSIFT
import math

def main():
	# Get the path of the training set
	parser = ap.ArgumentParser()
	parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
	args = vars(parser.parse_args())

	# Get the training classes names and store them in a list
	train_path = args["trainingSet"]
	#train_path = "dataset/train/"
	train(train_path)

def train(train_path):
	training_names = os.listdir(train_path)
	numWords = 1000
	# Get all the path to the images and save them in a list
	# image_paths and the corresponding label in image_paths
	image_paths = []
	for training_name in training_names:
	    image_path = os.path.join(train_path, training_name)
	    image_paths += [image_path]
	# Create feature extraction and keypoint detector objects
	fea_det = cv2.FeatureDetector_create("SIFT")
	des_ext = cv2.DescriptorExtractor_create("SIFT")
	# List where all the descriptors are stored
	des_list = []
	for i, image_path in enumerate(image_paths):
	    im = cv2.imread(image_path)
	    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	    print "Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths))
	    #kpts = fea_det.detect(im)
	    #kpts, des = des_ext.compute(im, kpts)
	    orb = cv2.ORB()
	    kpts, des = orb.detectAndCompute(im,None)
            draw_path="./draw_image/drawdes_"+os.path.basename(image_path)+".png";
            im_draw=cv2.drawKeypoints(im,kpts,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
            plt.axis("off")
            plt.imshow(im_draw)
            plt.savefig(draw_path,bbox_inches="tight")
	    # rootsift
	    #rs = RootSIFT()
	    #des = rs.compute(kpts, des)
	    des_list.append((image_path, des))

	# Stack all the descriptors vertically in a numpy array
	#downsampling = 1
	#descriptors = des_list[0][1][::downsampling,:]
	#for image_path, descriptor in des_list[1:]:
	#    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[1:]:
	    descriptors = np.vstack((descriptors, descriptor))

	# Perform k-means clustering
	print "Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0])

	#to float
	descriptors_float=[];
	for d in descriptors:
	    descriptors_float.append(map(float,d));

	voc, variance = kmeans(descriptors_float, numWords, 1)

	# Calculate the histogram of features
	im_features = np.zeros((len(image_paths), numWords), "float32")
	for i in xrange(len(image_paths)):
	    words, distance = vq(des_list[i][1],voc)
	    for w in words:
		im_features[i][w] += 1

	# Perform Tf-Idf vectorization
	nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
	idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

	# Perform L2 normalization
	im_features = im_features*idf
	im_features = preprocessing.normalize(im_features, norm='l2')
	joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)

if __name__=='__main__':
   main()
