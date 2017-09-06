'''
cite from "Deep Learning - The detail explaination and practice of classical model in Caffe"
'''

import numpy as np
import pandas as pd
import caffe

MODEL_FILE = '/examples/kaggle/facial/lenet_fk_deploy.prototxt'
PRETRAINED = '/examples/kaggle/facial/lenet/_iter_10000.caffemodel'

dataframe = pd.read_csv('/kaggle/facial/data/test.csv',header=0)
dataframe['Image'] = dataframe['Image'].apply(lambda im: np.fromstring(im, sep=' '))
data = np.vstack(dataframe['Image'].values)
data = data.reshape([-1,96,96])
data = data.astype(np.float32)

#scale between 0 and 1
data = data/255.
data = data.reshape([-1,1,96,96])

caffe.set_device(0)  # we can use 0,2,3
caffe.set_mode_gpu()

#net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
net=caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
caffe.set_mode_gpu()

total_images = data.shape[0]
s = np.shape(data)
print 'Total images to be predicted:', total_images
dataL = np.zeros([total_images,1,1,1], np.float32)

#read data to network
net.set_input_arrays(data.astype(np.float32),dataL.astype(np.float32))
pred = net.forward()

#change to be image coord
predicted = net.blobs['ip2'].data * 48 + 48

print 'Predicted', predicted

print 'Predicted shape:', predicted.shape
print 'Saving to scv..'

np.savetxt("testtest_fkp_lenet_output.csv", predicted, delimiter=",")

