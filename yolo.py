import numpy as np
import argparse
import cv2 as cv
import time
import os
from yolo_utils import infer_image, add_smoke
from PIL import Image

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
                        type=str,
                        default='./yolov3-coco/',
                        help='The directory where the model weights and \
			  configuration files are.')

    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/yolov3-custom_final.weights',
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

    parser.add_argument('--download-model',
                        type=bool,
                        default=False,
                        help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

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
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            co = 0
            while True:

                grabbed, frame = vid.read()
                co += 1

                # Checking if the complete video is read
                if not grabbed:
                    break
                
				
                if width is None or height is None:
                    height, width = frame.shape[:2]
                if(co%24==0):
                    frame, detect = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS, co)
                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                    writer.write(frame)
                    if detect:
                        for i in range(1,48):
                            grabbed,frame = vid.read()
                            co += 1
                            height, width = frame.shape[:2]
                            add_smoke(frame,height)
                            labelledImg = cv.imread("pasted_image.jpg")
                            if writer is None:
                                # Initialize the video writer
                                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                            writer.write(labelledImg)
                else:            

                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)
                    writer.write(frame)

            print("[INFO] Cleaning up...")
            writer.release()
            vid.release()

    else:
        print("Please enter the video path")
