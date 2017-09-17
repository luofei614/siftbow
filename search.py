#python search.py -i dataset/train/ukbench00000.jpg

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from pylab import *
#from PIL import Image
#from rootsift import RootSIFT

# Get the path of the training set
def main():
	parser = ap.ArgumentParser()
	parser.add_argument("-i", "--image", help="Path to query image", required="True")
	args = vars(parser.parse_args())
	# Get query image path
	image_path = args["image"]
	search(image_path);
def search(image_path):
	# Load the classifier, class names, scaler, number of clusters and vocabulary
	im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

	# Create feature extraction and keypoint detector objects
	fea_det = cv2.FeatureDetector_create("SIFT")
	des_ext = cv2.DescriptorExtractor_create("SIFT")

	# List where all the descriptors are stored
	des_list = []

	im = cv2.imread(image_path)
        #grey
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB()
	kpts, des = orb.detectAndCompute(im,None)

	#kpts = fea_det.detect(im)
	#kpts, des = des_ext.compute(im, kpts)

	# rootsift
	#rs = RootSIFT()
	#des = rs.compute(kpts, des)

	des_list.append((image_path, des))

	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]

	#
	test_features = np.zeros((1, numWords), "float32")
	print(descriptors)
	words, distance = vq(descriptors,voc)
	for w in words:
	    test_features[0][w] += 1

	# Perform Tf-Idf vectorization and L2 normalization
	test_features = test_features*idf
	test_features = preprocessing.normalize(test_features, norm='l2')

	score = np.dot(test_features, im_features.T)
	rank_ID = np.argsort(-score)
        result=[]
	for i, ID in enumerate(rank_ID[0][0:16]):
            result.append((image_paths[ID],'%.2f' % score[0][ID]))
        im_draw=cv2.drawKeypoints(im,kpts,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
        plt.subplot(2,1,1)
        plt.axis("off")
        plt.imshow(im_draw)
        plt.subplot(2,1,2)
        first_path="./draw_image/drawdes_"+os.path.basename(result[0][0])+".png";
	im_first = cv2.imread(first_path)
        plt.axis("off")
        plt.imshow(im_first)
        draw_path="./search_image/s_"+os.path.basename(image_path)+".png";
        result.append((draw_path,'0'));
        plt.savefig(draw_path,bbox_inches='tight', dpi = 200)

        print(result)
        return result

if __name__=='__main__':
     main();


'''
# Visualize the results
figure()
gray()
subplot(5,4,1)
imshow(im[:,:,::-1])
axis('off')
for i, ID in enumerate(rank_ID[0][0:16]):
    img = Image.open(image_paths[ID])
    gray()
    subplot(5,4,i+5)
    imshow(img)
    axis('off')

show()
'''
