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

from pylab import *
import cgi
import json
#from PIL import Image
#from rootsift import RootSIFT

# Get the path of the training set
'''
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required="True")
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]
'''
def search(image_path):
	# Load the classifier, class names, scaler, number of clusters and vocabulary
	im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

	# Create feature extraction and keypoint detector objects
	fea_det = cv2.FeatureDetector_create("SIFT")
	des_ext = cv2.DescriptorExtractor_create("SIFT")

	# List where all the descriptors are stored
	des_list = []

	im = cv2.imread(image_path)

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
        print('ids:')
        print(rank_ID)
        result=[]
	for i, ID in enumerate(rank_ID[0]):
            result.append(image_paths[ID])
        return result


def application(env,start_response):
    start_response('200 ok',[('Content-Type','text/html'),('Access-Control-Allow-Origin','*')]);
    lens=env.get('CONTENT_LENGTH','0')
    if lens=="":
        length=0;
    else:
        length=int(lens)
    body=''
    result=[]
    fields=cgi.FieldStorage(fp=env['wsgi.input'],environ=env,keep_blank_values=1)
    if 'search' in fields:
        result=search(fields.getvalue('search'))
    return [json.dumps(result)]


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
