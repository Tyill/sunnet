
from libskynet import*
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input


### Create net
net = snResNet50.createNet(snType.calcMode.CPU)

### Set weight
weightTF = snWeight.getResNet50Weights()

if (not snResNet50.setWeights(net, weightTF)):
    print('Error setWeights')
    exit(-1)

#################################

img_path = 'c:\\cpp\\other\\skyNet\\example\\resnet50\\images\\elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # (224,224,3)

x = np.moveaxis(x, -1, 1)

outAr = np.zeros((1, 1, 1, 1000), ctypes.c_float)


net.forward(False, x.copy(), outAr)

mx = np.argmax(outAr[0])

# for check: c:\cpp\other\skyNet\example\resnet50\imagenet_class_index.json
   
print('Predicted:', mx, 'val', outAr[0][0][0][mx])

for j in range(1000):
    print(j, ' val ', outAr[0][0][0][j])

#################################

# Compare with TF
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#
#model = ResNet50(weights='imagenet')
#
#
# img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#preds = model.predict(x)
#decode the results into a list of tuples (class, description, probability)
#(one such list for each sample in the batch)

#mx = np.argmax(preds[0])

#print('Predicted:', mx, 'val', preds[0][mx])

#print('Predicted:', decode_predictions(preds, top=3)[0])
#Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

#for j in range(1000):
#    print(j, ' val ', preds[0][j])



