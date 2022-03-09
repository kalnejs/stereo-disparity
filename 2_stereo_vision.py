import numpy as np
import cv2


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

def loop():
    cv_file = cv2.FileStorage()
    cv_file.open('camera_remap.xml', cv2.FileStorage_READ)

    RM1x = cv_file.getNode('RM1x').mat()
    RM1y = cv_file.getNode('RM1y').mat()
    RM2x = cv_file.getNode('RM2x').mat()
    RM2y = cv_file.getNode('RM2y').mat()

    cv_file.release()

    pipe_l = gstreamer_pipeline(sensor_id=1)
    pipe_r = gstreamer_pipeline(sensor_id=0)

    cap_l = cv2.VideoCapture(pipe_l, cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(pipe_r, cv2.CAP_GSTREAMER)                 

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)

    while True:

        succes_l, frame_l = cap_l.read()
        succes_r, frame_r = cap_r.read()

        if succes_l and succes_r:

            frame_l = cv2.remap(frame_l, RM1x, RM1y, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, 0)
            frame_r = cv2.remap(frame_r, RM2x, RM2y, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE, 0)

            for line in range(0, int(frame_l.shape[0] / 20)):
                frame_l_color = cv2.line(frame_l, (0, line*20), (frame_l.shape[1], line*20), (10, 255, 255))
                frame_r_color = cv2.line(frame_r, (0, line*20), (frame_r.shape[1], line*20), (10, 255, 255))


            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)


            stack = np.hstack((frame_l_color, frame_r_color))

            cv2.imshow("stack", stack)

            disparity = stereo.compute(gray_l, gray_r)/16

            min = disparity.min()
            max = disparity.max()

            disparity = np.uint8(255 * (disparity - min) / (max - min))

            cv2.imshow("disparity", disparity)

        keyCode = cv2.waitKey(1) & 0xFF

        if keyCode == 27:
            cap_l.release()
            cap_r.release()
            break

if __name__ == "__main__":
    loop()