import cv2
from box_utils import *
import uuid
import json
import time
import threading
import sys

import onnxruntime as ort

face_center_x = 0.5
face_center_y = 0.5
face_size = 0.1

exiting = False
visualize = True

for a in sys.argv[1:]:
    if str(a).__contains__("silent") or str(a).__contains__("quiet"):
        visualize = False

class Face:
    center_x = 0.5
    center_y = 0.5
    size = 0.1
    id = ''

    def __init__(self, id, center_x, center_y, size):
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.size = size

class FaceEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

last_frame_faces = []

def camera_loop():
    global last_frame_faces, exiting, face_center_x, face_center_y, face_size, app

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)
    #video_capture.set(cv2.CAP_PROP_FPS, 30)
    #video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    onnx_path = 'model640.onnx'
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    while True:
        starttime = time.time()
        ret, frame = video_capture.read()
        if frame is None:
            frame = cv2.imread('sample.jpg')
        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
            #img = cv2.resize(img, (640, 480)) # resize
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            this_frame_faces = []


            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                x1, y1, x2, y2 = box

                face_center_x = (int((x1 + x2) / 2) / frame.shape[1])
                face_center_y = (int((y1 + y2) / 2) / frame.shape[0])
                face_size = ((x2 - x1) / frame.shape[1])
                face_id = str(uuid.uuid4())[:8].upper()

                newface = Face(face_id, face_center_x, face_center_y, face_size)
                deadzone = 0.002
                trackingzone = 0.15

                for oldface in last_frame_faces:
                    xdif = abs(oldface.center_x - newface.center_x)
                    ydif = abs(oldface.center_y - newface.center_y)
                    sdif = abs(oldface.size - newface.size)

                    if xdif < trackingzone:
                        if ydif < trackingzone:
                            if sdif < trackingzone:
                                newface.id = oldface.id

                    if xdif < deadzone:
                        if ydif < deadzone:
                            if sdif < deadzone * 20:
                                newface = oldface
                                break
                    if newface.id == oldface.id:
                        newface.center_x = (newface.center_x + oldface.center_x*3) / 4
                        newface.center_y = (newface.center_y + oldface.center_y*3) / 4
                        newface.size = (newface.size + oldface.size*6) / 7

                if visualize:
                    cv2.circle(frame, (int(newface.center_x * frame.shape[1]), int(newface.center_y * frame.shape[0])),
                               int(newface.size * frame.shape[1]), (0, 0, 255), 5)

                    font = cv2.FONT_HERSHEY_DUPLEX
                    text = newface.id
                    cv2.putText(frame, text, (int(newface.center_x * frame.shape[1]*0.9), int(newface.center_y * frame.shape[0]*1.2)), font, 1, (0, 0, 255), 1)

                id_found = False
                for f in this_frame_faces:
                    if f.id == newface.id:
                        id_found = True
                        break
                if id_found == False:
                    this_frame_faces.append(newface)

            last_frame_faces = this_frame_faces

            if visualize:
                cv2.imshow('Face detector server - Hit Q to quit', frame)
        #print(time.time() - starttime)
        # Hit 'q' on the keyboard to quit!
        if exiting or cv2.waitKey(1) & 0xFF == ord('q'):
            exiting = True
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

from flask import Flask
app = Flask(__name__)

def run_flask():
    global exiting
    if __name__ == '__main__':
        time.sleep(5)
        app.run()

th = threading.Thread(target=camera_loop)
th.start()

@app.route('/')
def hello_world():
    global last_frame_faces
    return json.dumps(last_frame_faces, cls=FaceEncoder)

def main():
    fp = threading.Thread(target=run_flask)
    fp.start()

    th.join()
    print("Exiting")
    import os
    import signal

    sig = getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(os.getpid(), sig)

    exit(0)
    # camera_loop()

if __name__ == '__main__':
    main()

