# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import mediapipe as mp
import argparse
import time
import cv2
mp_face_detection = mp.solutions.face_detection

def detect_and_predict_mask(image, maskNet):
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        height, width, channels = image.shape
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        
        #height, width, channels = image.shape
        # Showing informations on the screen
        boxes = []
        locs = []
        preds = []
        a = []
        if results.detections:
            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box
                x=int(bbox.xmin * width)
                y=int(bbox.ymin * height)
                w=int(bbox.width * width)
                h=int(bbox.height * height)

                startX = x 
                startY = y 
                endX = x + w
                endY = y + h
                if startX<0:
                    startX=0
                        
                boxes.append([startX, startY, endX, endY])

            for i in range(len(boxes)):
                
                startX, startY, endX, endY = boxes[i]
                #print("face",startY,endY, startX,endX )
                face = image[startY:endY, startX:endX]
                if len(face) > 0:
                    #print(i,face)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)    
                    face = np.expand_dims(face, axis=0)

                    locs.append((startX, startY, endX, endY))
                # only make a predictions if at least one face was detected
                    preds = maskNet.predict(face)
                    a=np.append(a,preds)

	# return a 2-tuple of the face locations and their corresponding
	# locations
    return (locs, a)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector1.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    start_time = time.time()
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
    (locs, preds) = detect_and_predict_mask(frame, maskNet)
    #print("a",preds)
    # loop over the detected face locations and their corresponding
    # locations
    #print("pred",preds) 
    for i in range(len(locs)):
        print("i",i)

        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = locs[i]
        mask = preds[0+2*i]
        withoutMask = preds[1+2*i]
        
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    print("--- %s seconds ---" % (time.time() - start_time))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()