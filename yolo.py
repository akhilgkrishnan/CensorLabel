import numpy as np
import argparse
import cv2 as cv
import time
import os
from yolo_utils import infer_image, add_label
from PIL import Image
from pathlib import Path

FLAGS = []

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/helmet6000.weights',
                        help='Path to the file which contains the weights for YOLOv3.')

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
                        help='Path to the file having the labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.5,
                        help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.3,
                        help='The threshold to use when applying the Non-Max Suppresion')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')
    parser.add_argument("-u", "--use_gpu", 
                        type=bool, 
                        default=False,
	                    help="boolean indicating if CUDA GPU should be used.")
    parser.add_argument("-d", "--display", 
                        type=bool, 
                        default=False,
	                    help="Show frame display")                                        

    FLAGS, unparsed = parser.parse_known_args()
    
    startt = time.time()
    # Load the weights and configutation to form the pretrained YOLOv3 model for smoking detection
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    print("sucess")
   
    
    if FLAGS.use_gpu:
    	# set CUDA as the preferable backend and target
	    print("[INFO] setting preferable backend and target to CUDA...")
	    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
	    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    

    if FLAGS.video_path:
        frameCount = 0
        frame = cv.imread(FLAGS.video_path)
        height , width = frame.shape[:2]
        frame, detect = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)