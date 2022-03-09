import cv2
import pathlib
import os
import numpy as np
import glob

#https://github.com/JetsonHacksNano/CSI-Camera/blob/master/dual_camera.py
def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=1280/2,
        display_height=720/2,
        framerate=30,
        flip_method=0,
    ):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class StereoImageCapture():

    pipe_l = gstreamer_pipeline(sensor_id=1)
    pipe_r = gstreamer_pipeline(sensor_id=0)

    save_dir = "/capture"

    def __init__(self):
        self.count = 0
        self.check_dir()
        self.open_capture()
        self.capture_images()

    def check_dir(self):
        self.path = str(pathlib.Path(__file__).parent.resolve())+StereoImageCapture.save_dir

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        images = glob.glob(self.path+"/*.jpg")

        for image in images:
            os.remove(image)

    def open_capture(self):
        try:
            self.cap_l = cv2.VideoCapture(
                StereoImageCapture.pipe_l, cv2.CAP_GSTREAMER
            )
        except RuntimeError:
            print("Unable to open left capture")
            return

        try:
            self.cap_r = cv2.VideoCapture(
                StereoImageCapture.pipe_r, cv2.CAP_GSTREAMER
            )
        except RuntimeError:
            print("Unable to open right capture")
            return

    def capture_images(self):
        while(True):
            try:        
                ret_l, frame_l = self.cap_l.read() 
                ret_r, frame_r = self.cap_r.read()            

                if(ret_l and ret_r):
                    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                    
                    gray_stack = np.hstack((gray_l, gray_r)) 
                    cv2.imshow("capture", gray_stack)

            except RuntimeError:
                print("Could not read image from camera")


            keyCode = cv2.waitKey(1) & 0xFF

            if keyCode == 27:
                self.cap_l.release()
                self.cap_r.release()
                break

            if keyCode == 32:
                print("Saving {}".format(self.count))
                cv2.imwrite("{}/{}.jpg".format(self.path, self.count), gray_stack)
                self.count = self.count + 1

if __name__ == "__main__":
    stereo_capture = StereoImageCapture()