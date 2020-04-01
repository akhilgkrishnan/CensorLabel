import numpy as np
import argparse
import sys
import cv2 as cv
from PIL import Image
import time
import imutils
from yolo import yolo_detect
from Statutory import add_warning
import eel
from tkinter import * 
from tkinter import filedialog

eel.init('web') 

video_path = ''
@eel.expose
def btn_ResimyoluClick():
	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)
	global video_path
	video_path = filedialog.askopenfilename()
	print(video_path)
	return video_path

@eel.expose
def startLabel(movie_lang,gpu_support,display_frame):
	global video_path
	print(video_path,movie_lang,gpu_support,display_frame)
	eel.mSpinner()
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", type= str,default='./human-activity/resnet-34_kinetics.onnx',
		help="path to trained human activity recognition model")
	ap.add_argument("-c", "--classes", type=str,default='./human-activity/action_recognition_kinetics.txt',
		help="path to class labels file")
	ap.add_argument("-vo","--output",type=str,default="./output.avi",
		help="Video output name")	
	args = vars(ap.parse_args())

	# load the contents of the class labels file, then define the sample
	# duration (i.e., # of frames for classification) and sample size
	# (i.e., the spatial dimensions of the frame)
	CLASSES = open(args["classes"]).read().strip().split("\n")
	SAMPLE_DURATION = 32
	SAMPLE_SIZE = 112

	labels = ['smoking','drinking beer','driving car','driving tractor','riding a bike','riding scooter','smoking hookah','riding mountain bike','motorcycling']  

	# load the human activity recognition model
	print("[INFO] loading human activity recognition model...")
	neth = cv.dnn.readNet(args["model"])
	# Load the weights and configutation to form the pretrained YOLOv3 model for smoking detection
	nety = cv.dnn.readNetFromDarknet('./yolov3-coco/yolov3-custom.cfg', './yolov3-coco/helmet6000.weights')

	if gpu_support:
			print("[INFO] setting preferable backend and target to CUDA...")
			neth.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
			neth.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
			print("[INFO] setting preferable backend and target to CUDA...")
			nety.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
			nety.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
			
	def activity_detect(frames):
		# now that our frames array is filled we can construct our blob
		blob = cv.dnn.blobFromImages(frames, 1.0,
			(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
			swapRB=True, crop=True)
		blob = np.transpose(blob, (1, 0, 2, 3))
		blob = np.expand_dims(blob, axis=0)
		# pass the blob through the network to obtain our human activity
		# recognition predictions
		neth.setInput(blob)
		start = time.time()
		outputs = neth.forward()
		end = time.time()
		print("Time taken is ",end-start)
		return CLASSES[np.argmax(outputs)]


	# grab a pointer to the input video stream
	print("[INFO] accessing video stream...")
	vid = cv.VideoCapture(video_path)
	writer = None
	fps = vid.get(cv.CAP_PROP_FPS)
	print("Fps is :",fps)
	# loop until we explicitly break from it
	while True:
		# initialize the batch of frames that will be passed through the
		# model
		frames = []

		# loop over the number of required sample frames
		for i in range(0, SAMPLE_DURATION):
			# read a frame from the video stream
			grabbed, frame = vid.read()
            

			# if the frame was not grabbed then we've reached the end of
			# the video stream so exit the script
			if not grabbed:
				break
		
			# otherwise, the frame was read so resize it and add it to
			# our frames list
			frame = imutils.resize(frame, width=400)
			frames.append(frame)
		

		#return label
		if(len(frames)>31):
			firstLabel = activity_detect(frames[:16])
			secondLabel = activity_detect(frames[16:])

		if firstLabel == secondLabel:
			label = firstLabel
			print("label is :",label)
			if label in labels:
				detect = yolo_detect(frames,writer,label,nety,fps)
				print("detect is",detect)
				if detect == 1:
					for i in range(0,84):
						grabbed, frame = vid.read()
						frames.append(frame)
						if not grabbed:
							break
                        
      
                        
					for frame in frames:
						cv.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
						cv.putText(frame, firstLabel, (10, 25), cv.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
						add_warning(frame,frame.shape[0],"Images/statutory/smoke.png")
						frame = cv.imread("pasted_image.jpg")
						if writer is None:
							fourcc = cv.VideoWriter_fourcc(*"MJPG")
							writer = cv.VideoWriter(args["output"], fourcc, fps,(frame.shape[1], frame.shape[0]), True)
						writer.write(frame)
				else:
					
					for frame in frames :

					    # cv.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                        # cv.putText(frame, firstlabel, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)	
						if writer is None:
							fourcc = cv.VideoWriter_fourcc(*"MJPG")
							writer = cv.VideoWriter(args["output"], fourcc, fps,(frame.shape[1], frame.shape[0]), True)
						writer.write(frame)		
		else:
			for frame in frames:
				cv.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
				cv.putText(frame, firstLabel, (10, 25), cv.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
				if writer is None:
					# Initialize the video writer 
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter(args["output"], fourcc, fps,(frame.shape[1], frame.shape[0]), True)
				writer.write(frame)						
	print("[INFO] Cleaning up...")
	eel.mSpinner()
	eel.mAddTick()
	writer.release()
	vid.release()
eel.start('main2.html', size=(700, 600))
	
		
