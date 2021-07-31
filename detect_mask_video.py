# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import multiprocessing 
from playsound import playsound
import tkinter
from tkinter import messagebox
import smtplib

from termcolor import colored
root = tkinter.Tk()
root.withdraw()

count=0

def detect_and_predict_mask(frame, faceNet, maskNet):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]
		
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	
	return (locs, preds)

#Program start
if __name__ == '__main__': 
		
	prototxtPath = r"face_detector\deploy.prototxt"
	weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	maskNet = load_model(r"D:\B.Tech\TY\SEM1\Mini Project\x\mask_detector.model")
	
	SUBJECT = "Subject"
	TEXT = "One Visitor violated Face Mask Policy.See in the camera to recognize user." \
		"A Person has been detected without Mask"

	print("\n\n\n\nWELCOME\n\nFace Mask Detector:")
	# initialize the video stream
	print("\n[INFO] Starting Webcam:")
	vs = VideoStream(src=0).start()

	# loop over the frames from the video stream
	while True:
		
		frame = vs.read()
		frame = imutils.resize(frame,width=800)
		
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		for (box, pred) in zip(locs, preds):
			
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
		
			label = "Mask" if mask > withoutMask else "Without_Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
			"""if mask < withoutMask:
				count+=1
				messagebox.showerror("Warning ","Access Denied.\nPlease wear a Face Mask")
				message = 'Subject: {}\n\n{}'.format(SUBJECT,TEXT)
				mail = smtplib.SMTP('smtp.gmail.com',587)
				mail.ehlo()
				mail.starttls()
				mail.login('demo6946@gmail.com','Demo@1236946')
				mail.sendmail('demo6946@gmail.com','demo6946@gmail.com',message)
				mail.close()"""
			
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
			cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# show the output frame
		cv2.imshow("Live Video : Webcam", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord('q'):
			break
	#print("\nTotal Number of violators = "+str(count))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()