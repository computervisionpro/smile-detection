import numpy as np
import argparse
import imutils
import cv2
import dlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import face_utils

# python 15.2.smile-video.py -m ./data2/lsmile.hdf5 -v ./data/cr7.mp4


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to smile detection model')
ap.add_argument('-v','--video', help='path to optional video')

args = vars(ap.parse_args())

# load the model
model = load_model(args['model'])

#get face detector
detector = dlib.get_frontal_face_detector()

if not args.get('video',False):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args['video'])


while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = imutils.resize(frame, width=500)
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print('[INFO]: Detection')
    rects = detector(gray,0)
    #print(rects)

    for (i ,rect) in enumerate(rects):
        (x,y,w,h) = face_utils.rect_to_bb(rect)
        gr_roi = gray[y:y+h, x:x+w]
        #print(gr_roi)
            
        # resizing input acc. to the model
        gr_roi = cv2.resize(gr_roi, (28,28))
        gr_roi = gr_roi.astype('float')/255.0

        gr_roi = img_to_array(gr_roi) # (28,28,1) dtype- f32
        gr_roi = np.expand_dims(gr_roi, axis=0) # (1,gr_roi.shape)
        
        # calculate the prob of both smiling or not smiling
        (notSmiling, smiling) = model.predict(gr_roi)[0]
        print('[INFO]: Prediction')
        label = "Smiling" if smiling>notSmiling else "Not Smiling"

        # display the output
        
        cv2.putText(frame_copy, label, (x,y-10), cv2.FONT_ITALIC, 0.5, 
                            (0,255,255),2)
        cv2.rectangle(frame_copy, (x,y), (x+w, y+h), (255,255,0),1)

    frame_copy = imutils.resize(frame_copy, width=900)
    
    cv2.imshow("Smile please", frame_copy)
    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


