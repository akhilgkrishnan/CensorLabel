import numpy as np
import argparse
import cv2 as cv
import time
import os
from yolo_utils import infer_image
from pathlib import Path

FLAGS = []

def yolo_detect(frames,writer,labelh,net,fps):
    
    
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

    FLAGS, unparsed = parser.parse_known_args()
    
  
    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')
    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
   
        
    #Extracting the audio file from the video file
    #os.system('ffmpeg -i '+FLAGS.video_path+' -ab 160k -ac 2 -ar 44100 -vn Audio/'+Path(FLAGS.video_path).stem+'-audio.wav')
    height , width =  None, None
    writer = None

    for frame in frames:
      
        if width is None or height is None:
            width = frame.shape[1]
            height  = frame.shape[0]
        
        detect = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS,labelh)
        if detect == 0:
            continue
        else:
            return detect            
    #Binding the audio file to the output.avi file
    #os.system('ffmpeg -i output.avi -i Audio/'+Path(FLAGS.video_path).stem+'-audio.wav -c copy Video/'+Path(FLAGS.video_path).stem+'-Ouput.mkv')

 
