from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array  # tol list
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
from lenet import LeNet
import os

# python 15.SMILE.py -d ./data2/SMILEsmileD-master/SMILEs --model ./my/lenet-smile.hdf5

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
help="path to input dataset of faces")
ap.add_argument("-m","--model",required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize the list of data and labels
data=[]
labels=[]

# loop over the input
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    #load image. pre-process it and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image,width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path
    label = imagePath.split(os.path.sep)[-3]
    # os.path.sep = /
    # imagePath = str
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# scaling the raw pixel
data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

# converting labels to vectors
le = LabelEncoder().fit(labels)

# one hot-encoding
labels = to_categorical(le.transform(labels),2)

# calculating weights for imbalance
classTotals = labels.sum(axis=0)
classWeight = classTotals.max()/classTotals

# split the data
(trainX, testX, trainY, testY) = train_test_split(data,labels,
test_size=0.2, stratify=labels,random_state=42)

print('Train x', trainX.shape)
print('Train y', trainY.shape)
print('Test x', testX.shape)
print('Test y', testY.shape)

# initiate the model
print("[INFO] compiling model")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])

# train the network
print("[INFO] training network")
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)

print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,15),H.history['loss'], label='train_loss')
plt.plot(np.arange(0,15),H.history['val_loss'], label='validation_loss')
plt.plot(np.arange(0,15),H.history['acc'], label='train_accuracy')
plt.plot(np.arange(0,15),H.history['val_acc'], label='validation_accuracy')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.show()
