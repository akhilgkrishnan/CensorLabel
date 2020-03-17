import numpy as np
import argparse
import cv2 as cv
import time
import os
<<<<<<< HEAD
from yolo_utils import infer_image, add_label
=======
from yolo_utils import infer_image, add_smoke
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
from PIL import Image

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

<<<<<<< HEAD

    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/yolov3hs.weights',
=======
    parser.add_argument('-m', '--model-path',
                        type=str,
                        default='./yolov3-coco/',
                        help='The directory where the model weights and \
			  configuration files are.')

    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/yolov3-custom_final.weights',
<<<<<<< HEAD
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
=======
>>>>>>> parent of 4097696... add fps
                        help='Path to the file which contains the weights \
			 	for YOLOv3.')

    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='./yolov3-coco/yolov3-custom.cfg',
                        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-v', '--video-path',
                        type=str,
                        help='The path to the video file')

    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='./output.avi',
                        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
                        type=str,
                        default='./yolov3-coco/coco-labels',
                        help='Path to the file having the \
					labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.5,
                        help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.3,
                        help='The threshold to use when applying the \
				Non-Max Suppresion')

<<<<<<< HEAD
=======
    parser.add_argument('--download-model',
                        type=bool,
                        default=False,
                        help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')
<<<<<<< HEAD
    parser.add_argument("-u", "--use_gpu", 
                        type=bool, 
                        default=False,
	                    help="boolean indicating if CUDA GPU should be used")                    

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.use_gpu:
    	# set CUDA as the preferable backend and target
	    print("[INFO] setting preferable backend and target to CUDA...")
	    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
	    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    #Extracting the audio file from the video file
    os.system('ffmpeg -i '+FLAGS.video_path+' -ab 160k -ac 2 -ar 44100 -vn audio.wav')

=======

    FLAGS, unparsed = parser.parse_known_args()

>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model for smoking detection
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1]
                   for i in net.getUnconnectedOutLayers()]

    if FLAGS.video_path:
<<<<<<< HEAD
        
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            fps = vid.get(cv.CAP_PROP_FPS)
            print("FPS is :",fps)
            height , width = None, None
=======
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            height, width = None, None
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

        finally:
<<<<<<< HEAD
            frameCount = 0
=======
            co = 0
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
            while True:

                grabbed, frame = vid.read()
                
<<<<<<< HEAD
                print("Frame count",frameCount)
=======
                print("Frame count",co)
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9

                # Checking if the complete video is read
                if not grabbed:
                    break
                
				
                if width is None or height is None:
                    height, width = frame.shape[:2]
                #Take first frame from each second for detection    
<<<<<<< HEAD
<<<<<<< HEAD
                if(frameCount%fps==0):
=======
                if(co%1==0):
>>>>>>> parent of 4097696... add fps
                    
                    frame, detect = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS, frameCount)
                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                    writer.write(frame)
                    #Check the frame contain any detection, if detection is occur, label statutory warning on the next 120 frames
                    if(detect==1):
                        for i in range(1,120):
                            grabbed,frame = vid.read()
                            frameCount += 1
                            print("Frame count",frameCount)
                            height, width = frame.shape[:2]
                            add_label(frame,height,'smoke.png')
=======
                if(co%1==0):
                    
                    frame, detect = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS, co)
                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                    writer.write(frame)
                    #Check the frame contain any detection, if detection is occur, label statutory warning on the next 120 frames
                    if detect:
                        for i in range(1,120):
                            grabbed,frame = vid.read()
                            co += 1
                            print("Frame count",co)
                            height, width = frame.shape[:2]
                            add_smoke(frame,height)
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
                            labelledImg = cv.imread("pasted_image.jpg")
                            if writer is None:
                                # Initialize the video writer
                                fourcc = cv.VideoWriter_fourcc(*"MJPG")
<<<<<<< HEAD
                                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, fps,
                                                (frame.shape[1], frame.shape[0]), True)
                            writer.write(labelledImg)
                    elif(detect==2):
                        for i in range(1,120):
                            grabbed,frame = vid.read()
                            frameCount += 1
                            print("Frame count",frameCount)
                            height, width = frame.shape[:2]
                            add_label(frame,height,'helmet.png')
                            labelledImg = cv.imread("pasted_image.jpg")
                            if writer is None:
                                # Initialize the video writer
                                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                            writer.write(labelledImg)        
=======
                                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                            writer.write(labelledImg)
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
                else:            

                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
<<<<<<< HEAD
<<<<<<< HEAD
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, fps,
=======
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
>>>>>>> parent of 4097696... add fps
                                                (frame.shape[1], frame.shape[0]), True)
                    writer.write(frame)
                frameCount += 1    
=======
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                    writer.write(frame)
                co += 1    
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9

            print("[INFO] Cleaning up...")
            writer.release()
            vid.release()

<<<<<<< HEAD
            #Binding the audio file to the output.avi file
            os.system('ffmpeg -i output.avi -i audio.wav -c copy output.mkv')

=======
>>>>>>> ca3a2cca6597fb836eaf2963885ff7fbc23995c9
    else:
        print("Please enter the video path..")
