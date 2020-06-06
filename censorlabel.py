import numpy as np
import argparse
import cv2 as cv
import sys
import time
import imutils
from yolo import yolo_detect
from Statutory import add_warning
import eel
from tkinter import * 
from tkinter import filedialog
from pathlib import Path
import os

eel.init('web') 
writer = None
video_path = ''
@eel.expose
def btn_ResimyoluClick():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    global video_path
    video_path = filedialog.askopenfilename(filetypes = (("mp4 files","*.mp4"),("mpv files","*.mpv"),("all files","*.*")))
    print(video_path)
    return video_path

@eel.expose
def cancel():
    sys.exit(0)

@eel.expose
def startLabel(movie_lang,gpu_support,display_frame):
    global video_path
    if video_path != '':
        eel.mSpinner()
        eel.info("Movie statutory labeling started")
    else:
        eel.info("select video path")     
    try:
        
        os.system('ffmpeg -i '+video_path+' -ab 160k -ac 2 -ar 44100 -vn Audio/'+Path(video_path).stem+'-audio.wav')
        print(video_path,movie_lang,gpu_support,display_frame)
        
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

        labels = ['tasting beer','smoking','drinking beer','driving car','driving tractor','riding a bike','riding scooter','smoking hookah','riding mountain bike','motorcycling']
        riding = ['motorcycling', 'riding a bike', 'riding scooter', 'riding mountain bike']
        smoking = ['smoking', 'smoking hookah']
        alcohol = ['tasting beer','drinking beer']
        driving = ['driving car','driving tractor']
        # load the human activity recognition model
        print("[INFO] loading human activity recognition model...")
        neth = cv.dnn.readNet(args["model"])
        # Load the weights and configutation to form the pretrained YOLOv3 model for smoking detection
        nethelmet = cv.dnn.readNetFromDarknet('./yolov3-coco/yolov3-custom.cfg', './yolov3-coco/helmet6000.weights')
        netsmoking = cv.dnn.readNetFromDarknet('./yolov3-coco/yolov3-custom.cfg', './yolov3-coco/yolosmoking.weights')
        netseatbelt = cv.dnn.readNetFromDarknet('./yolov3-coco/yolov3-custom1.cfg', './yolov3-coco/yoloseatbelt.weights')

        if gpu_support:
                print("[INFO] setting preferable backend and target to CUDA...")
                neth.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                neth.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
                print("[INFO] setting preferable backend and target to CUDA...")
                nethelmet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                nethelmet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

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
            outputs = neth.forward()
            z = outputs.argsort()[-5:][0][-5:]
            j = [CLASSES[x] for x in z]
            print(j)
            print(np.argmax(outputs))
            return CLASSES[np.argmax(outputs)]

        def writeFrame(frame,fps):
            global writer
            if writer is None:
                # Initialize the video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)
            
        # grab a pointer to the input video stream
        print("[INFO] accessing video stream...")
        vid = cv.VideoCapture(video_path)
        fps = vid.get(cv.CAP_PROP_FPS)
        print("Fps is :",fps)
        firstLabel = ''
        secondLabel = ''
        thirdLabel = ''
    
        # loop until we explicitly break from it
        while True:
            # initialize the batch of frames that will be passed through the
            # model
            frames = []
            # loop over the number of required sample frames
            for i in range(0, SAMPLE_DURATION):
                # read a frame from the video stream
                (grabbed, frame) = vid.read()
                # if the frame was not grabbed then we've reached the end of
                # the video stream so exit the script
                if not grabbed:
                    break
                # otherwise, the frame was read so resize it and add it to
                # our frames list
                #frame = imutils.resize(frame, width=400)
                frames.append(frame)
            
            if(len(frames)>31):
                firstLabel = activity_detect(frames[:16])
                secondLabel = activity_detect(frames[16:])
                print(firstLabel)
                print(secondLabel)
                
            else:
                for frame in frames:
                    writeFrame(frame,fps)
                break
            
            if (firstLabel == secondLabel) or (firstLabel == thirdLabel) or (firstLabel in alcohol) or (secondLabel in alcohol) or (firstLabel in smoking) or (secondLabel in smoking):
                thirdLabel = secondLabel
                print(thirdLabel)
                label = firstLabel
                if (label in riding):
                    detect = yolo_detect(frames,label,nethelmet)
                    if detect == 1:
                        eel.info("Riding without helmet detected")
                        for i in range(0,130):
                            (grabbed, frame) = vid.read()
                            if not grabbed:
                                break
                            frames.append(frame) 
                        for frame in frames:
                            frame = add_warning(frame,'Images/statutory/'+movie_lang+'/helmet.png')
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF
                            writeFrame(frame,fps)
                    else:
                        for frame in frames:
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF
                            writeFrame(frame,fps)        
                elif (firstLabel in smoking) or (secondLabel in smoking):
                    detect = yolo_detect(frames,label,netsmoking)
                    print("detect is :",detect)            
                    if detect == 2:
                        eel.info("Smoking detected")
                        for i in range(0,84):
                            (grabbed, frame) = vid.read()
                            if not grabbed:
                                break
                            frames.append(frame)
                        for frame in frames:
                            frame = add_warning(frame,'Images/statutory/'+movie_lang+'/smoke.png')
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF
                            writeFrame(frame,fps)
                    else:
                        for frame in frames:
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF
                            writeFrame(frame,fps)
                elif label in alcohol:
                    detect = yolo_detect(frames,label,netsmoking)
                    for i in range(0,84):
                        (grabbed, frame) = vid.read()
                        if not grabbed:
                            break
                        frames.append(frame)
                    if detect == 2:
                        eel.info("alcohol & smoking detected")
                        for frame in frames:
                            frame = add_warning(frame,'Images/statutory/'+movie_lang+'/smokealcohol.png')
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF    
                            writeFrame(frame,fps)
                    else:        
                        for frame in frames:
                            frame = add_warning(frame,'Images/statutory/'+movie_lang+'/alcohol.png')
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF    
                            writeFrame(frame,fps)
                elif label in driving:
                    detect = yolo_detect(frames,label,netseatbelt)
                    print("detect is",detect)
                    if detect == 3:
                        eel.info("driving without seatbelt detection")
                        for i in range(0,84):
                            (grabbed, frame) = vid.read()
                            if not grabbed:
                                break
                            frames.append(frame)
                        for frame in frames:
                            frame = add_warning(frame,'Images/statutory/'+movie_lang+'/seatbelt.png')
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF    
                            writeFrame(frame,fps)
                    else:
                        for frame in frames:
                            if display_frame:
                                cv.imshow("Statutory Labeling", frame)
                                key = cv.waitKey(1) & 0xFF
                            writeFrame(frame,fps)                         
                else:
                    for frame in frames:
                        if display_frame:
                            cv.imshow("Statutory Labeling", frame)
                            key = cv.waitKey(1) & 0xFF
                        writeFrame(frame,fps)                   
            else:
                for frame in frames:
                    if display_frame:
                        cv.imshow("Statutory Labeling", frame)
                        key = cv.waitKey(1) & 0xFF
                    writeFrame(frame,fps)
        
        eel.mSpinner()
        if video_path != '':
            eel.mAddTick()
         
        writer.release()
        vid.release()
        if display_frame:
            cv.destroyWindow("Statutory Labeling") 
        eel.info('Output file is saved to: Video/'+Path(video_path).stem+'-Ouput.mkv')
        os.system('ffmpeg -i output.avi -i Audio/'+Path(video_path).stem+'-audio.wav -c copy Video/'+Path(video_path).stem+'-Ouput.mkv')
        print('Output file is saved to: Video/'+Path(video_path).stem+'-Ouput.mkv')
        print("Process finished")
        os.system('rm Audio/'+Path(video_path).stem+'-audio.wav')
    except:
        print("An error occured")
        eel.mSpinner()
        eel.mAddCross()
        eel.info("An error occured")
            
eel.start('main2.html', size=(800, 600))


