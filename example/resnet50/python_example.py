
from libsunnetimport*
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input


### Create net
net = snResNet50.createNet()

### Set weight
weightTF = snWeight.getResNet50Weights()

if (not snResNet50.setWeights(net, weightTF)):
    print('Error setWeights')
    exit(-1)

#################################

img_path = 'c:\\cpp\\other\\sunnet\\example\\resnet50\\images\\elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # (224,224,3)

x = np.moveaxis(x, -1, 1)

outAr = np.zeros((1, 1, 1, 1000), ctypes.c_float)

import time

for i in range(100):
 ct = time.time()
 net.forward(False, x.copy(), outAr)
 print(time.time() - ct)

 mx = np.argmax(outAr[0])

 # for check: c:\cpp\other\sunnet\example\resnet50\imagenet_class_index.json
 print('Predicted:', mx, 'val', outAr[0][0][0][mx])




