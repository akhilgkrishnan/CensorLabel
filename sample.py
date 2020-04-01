

import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

version = '9 Feb 2020'

# prototxt = r'model/colorization_deploy_v2.prototxt'
# model = r'model/colorization_release_v2.caffemodel'
# points = r'model/pts_in_hull.npy'
# points = os.path.join(os.path.dirname(__file__), points)
# prototxt = os.path.join(os.path.dirname(__file__), prototxt)
# model = os.path.join(os.path.dirname(__file__), model)
# if not os.path.isfile(model):
#     sg.popup_scrolled('Missing model file', 'You are missing the file "colorization_release_v2.caffemodel"',
#                       'Download it and place into your "model" folder', 'You can download this file from this location:\n', r'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1')
#     exit()
# net = cv2.dnn.readNetFromCaffe(prototxt, model)     # load model from disk
# pts = np.load(points)

# # add the cluster centers as 1x1 convolutions to the model
# class8 = net.getLayerId("class8_ab")
# conv8 = net.getLayerId("conv8_313_rh")
# pts = pts.transpose().reshape(2, 313, 1, 1)
# net.getLayer(class8).blobs = [pts.astype("float32")]
# net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_filename=None, cv2_frame=None):
    """
    Where all the magic happens.  Colorizes the image provided. Can colorize either
    a filename OR a cv2 frame (read from a web cam most likely)
    :param image_filename: (str) full filename to colorize
    :param cv2_frame: (cv2 frame)
    :return: Tuple[cv2 frame, cv2 frame] both non-colorized and colorized images in cv2 format as a tuple
    """


# --------------------------------- The GUI ---------------------------------

# First the window layout...2 columns

left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')],
            [sg.Text('Version ' + version, font='Courier 8')]]

images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
              [sg.Button('Colorize Photo', key='-PHOTO-'), sg.Button('Start Webcam', key='-WEBCAM-'), sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
              [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],]
# ----- Full layout -----
layout = [[sg.Column(left_col), sg.VSeperator(), sg.Column(images_col)]]

# ----- Make the window -----
window = sg.Window('Photo Colorizer', layout, grab_anywhere=True)

# ----- Run the Event Loop -----
colorized = cap = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        # get list of files in folder
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':    # A file was chosen from the listbox
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            image = cv2.imread(filename)
            imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
            window['-IN-'].update(data=imgbytes_in)
            window['-OUT-'].update(data='')
            window['-IN FILE-'].update('')

            image, colorized = colorize_image(filename)
            imgbytes_out = cv2.imencode('.png', colorized)[1].tobytes()
            window['-OUT-'].update(data=imgbytes_out)
        except:
            continue
    elif event == '-PHOTO-':        # Colorize photo button clicked
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            image, colorized = colorize_image(filename)
            imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
            imgbytes_out = cv2.imencode('.png', colorized)[1].tobytes()
            window['-IN-'].update(data=imgbytes_in)
            window['-OUT-'].update(data=imgbytes_out)
        except:
            continue
    elif event == '-IN FILE-':      # A single filename was chosen
        filename = values['-IN FILE-']
        try:
            image = cv2.imread(filename)
            imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
            window['-IN-'].update(data=imgbytes_in)
            window['-OUT-'].update(data='')
        except:
            continue
    elif event == '-WEBCAM-':       # Webcam button clicked
        sg.popup_quick_message('Starting up your Webcam... this takes a moment....', auto_close_duration=1,  background_color='red', text_color='white', font='Any 16')
        window['-WEBCAM-'].update('Stop Webcam', button_color=('white','red'))
        cap = cv2.VideoCapture(0) if not cap else cap
        while True:                 # Loop that reads and shows webcam until stop button
            ret, frame = cap.read()     # Read a webcam frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert webcam frame to grayscale
            gray_3_channels = np.zeros_like(frame)          # Convert grayscale frame (single channel) to 3 channels
            gray_3_channels[:, :, 0] = gray
            gray_3_channels[:, :, 1] = gray
            gray_3_channels[:, :, 2] = gray
            image, colorized = colorize_image(cv2_frame=gray_3_channels)    # Colorize the 3-channel grayscale frame
            imgbytes_in = cv2.imencode('.png', gray_3_channels)[1].tobytes()
            imgbytes_out = cv2.imencode('.png', colorized)[1].tobytes()
            window['-IN-'].update(data=imgbytes_in)
            window['-OUT-'].update(data=imgbytes_out)
            event, values = window.read(timeout=0)  # Update the window outputs and check for new events
            if event in (None, '-WEBCAM-', 'Exit'): # Clicked the Stop Webcam button or closed window entirely
                window['-WEBCAM-'].update('Start Webcam', button_color=sg.theme_button_color())
                window['-IN-'].update('')
                window['-OUT-'].update('')
                break
    elif event == '-SAVE-' and colorized is not None:   # Clicked the Save File button
        filename = sg.popup_get_file('Save colorized image.\nColorized image be saved in format matching the extension you enter.')
        try:
            if filename:
                cv2.imwrite(filename, colorized)
                sg.popup_quick_message('Image save complete', background_color='red', text_color='white', font='Any 16')
        except:
            sg.popup_quick_message('ERROR - Image NOT saved!', background_color='red', text_color='white', font='Any 16')
# ----- Exit program -----
window.close()


