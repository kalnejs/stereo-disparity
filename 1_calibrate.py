import cv2
import pathlib
import os
import numpy as np
import glob

class StereoImageCalibrate():

    chessboard_size = (9,6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    square_size_mm = 29
    save_dir = "/capture"

    def __init__(self):
        self.objp = np.zeros((StereoImageCalibrate.chessboard_size[0]*StereoImageCalibrate.chessboard_size[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:StereoImageCalibrate.chessboard_size[0],0:StereoImageCalibrate.chessboard_size[1]].T.reshape(-1,2)

        self.objp = self.objp * StereoImageCalibrate.square_size_mm

        self.objpoints = [] 
        self.imgpoints1 = [] 
        self.imgpoints2 = []

        self.path = str(pathlib.Path(__file__).parent.resolve())+StereoImageCalibrate.save_dir 

        self.read_images()
        self.calibrate_each()
        self.calibrate_stereo()
        self.test_stereo()

    def read_images(self):
        images = glob.glob(self.path+"/*.jpg")

        count = 0
        dims = (0,0)        
        for img in images:
            frame = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            frame1, frame2 = np.hsplit(frame, 2)
            dimension = frame1.shape[::-1]
            count = count + 1

        print("Total images: {}".format(count))
        print("Dimension: {}".format(dimension))

        self.image_dims = dimension

        for img in images:

            frame = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            frame1, frame2 = np.hsplit(frame, 2)

            ret1, corners1 = cv2.findChessboardCorners(frame1, StereoImageCalibrate.chessboard_size, None)    
            ret2, corners2 = cv2.findChessboardCorners(frame2, StereoImageCalibrate.chessboard_size, None)

            if ret1 and ret2:
                self.objpoints.append(self.objp)

                corners1 = cv2.cornerSubPix(frame1, corners1, (11,11), (-1,-1), StereoImageCalibrate.criteria)
                self.imgpoints1.append(corners1)

                corners2 = cv2.cornerSubPix(frame2, corners2, (11,11), (-1,-1), StereoImageCalibrate.criteria)
                self.imgpoints2.append(corners2)

    def calibrate_each(self):
        ret1, self.M1, self.D1, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints1, self.image_dims, None, None)
        self.M1, roi1 = cv2.getOptimalNewCameraMatrix(self.M1, self.D1, self.image_dims, alpha=1, newImgSize=self.image_dims)

        ret2, self.M2, self.D2, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints2, self.image_dims, None, None)
        self.M2, roi2 = cv2.getOptimalNewCameraMatrix(self.M2, self.D2, self.image_dims, alpha=1, newImgSize=self.image_dims)

    def calibrate_stereo(self):



        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_FIX_TANGENT_DIST
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH 
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        
        ret, self.M1, self.D1, self.M2, self.D2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, self.imgpoints1, self.imgpoints2, 
                                                                                    self.M1, self.D1, self.M2, self.D2, 
                                                                                    self.image_dims, 
                                                                                    flags=flags,
                                                                                    criteria=StereoImageCalibrate.stereo_criteria)


        # print("M1:", self.M1)
        # print("....")
        # print("M2:", self.M2)
        # print("....")
        # print("D1:", self.D1)
        # print("....")
        # print("D2:", self.D2) 
        # print("....")
        # print("R:", R) 
        # print("....")
        # print("T:", T) 
        # print("....")

        cv_file = cv2.FileStorage('camera_intrinsics.xml', cv2.FILE_STORAGE_WRITE)

        cv_file.write('M1', self.M1)
        cv_file.write('D1', self.D1)
        cv_file.write('M2', self.M2)
        cv_file.write('D2', self.D2)

        cv_file.release()

        # cv_file = cv2.FileStorage('camera_extrinsics.xml', cv2.FILE_STORAGE_WRITE)

        # cv_file.write('R', R)
        # cv_file.write('T', T)

        # cv_file.release()

        self.R1, self.R2, self.P1, self.P2, Q, self.ROI1, self.ROI2 = cv2.stereoRectify(self.M1, self.D1, self.M2, self.D2, 
                                                                                        self.image_dims, R, T, 
                                                                                        # flags=cv2.CALIB_ZERO_DISPARITY,
                                                                                        flags=0,
                                                                                        alpha=1, 
                                                                                        newImageSize=(0,0))

        # print("R1:", self.R1)
        # print("....")
        # print("R2:", self.R2) 
        # print("....")

        # print("P1:", self.P1)
        # print("....")
        # print("P2:", self.P2) 
        # print("....")

        cv_file = cv2.FileStorage('camera_extrinsics.xml', cv2.FILE_STORAGE_WRITE)

        cv_file.write('R1', self.R1)
        cv_file.write('R2', self.R2)
        cv_file.write('P1', self.P1)
        cv_file.write('P2', self.P2)

        cv_file.release()


        self.RM1x, self.RM1y = cv2.initUndistortRectifyMap(self.M1, self.D1, self.R1, self.P1, self.image_dims, cv2.CV_32FC1)
        self.RM2x, self.RM2y = cv2.initUndistortRectifyMap(self.M2, self.D2, self.R2, self.P2, self.image_dims, cv2.CV_32FC1)


        cv_file = cv2.FileStorage('camera_remap.xml', cv2.FILE_STORAGE_WRITE)

        cv_file.write('RM1x', self.RM1x)
        cv_file.write('RM1y', self.RM1y)
        cv_file.write('RM2x', self.RM2x)
        cv_file.write('RM2y', self.RM2y)

        cv_file.release()

    def test_stereo(self):

        cv_file = cv2.FileStorage('camera_remap.xml', cv2.FILE_STORAGE_READ)


        RM1x = cv_file.getNode("RM1x").mat()
        RM1y = cv_file.getNode("RM1y").mat()
        RM2x = cv_file.getNode("RM2x").mat()
        RM2y = cv_file.getNode("RM2y").mat()

        cv_file.release()

        images = glob.glob(self.path+"/*.jpg")
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)

        for img in images:

            frame = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            frame_l, frame_r = np.hsplit(frame, 2)



            frame_l = cv2.remap(frame_l, RM1x, RM1y, cv2.INTER_LINEAR , cv2.BORDER_CONSTANT, 0)
            frame_r = cv2.remap(frame_r, RM2x, RM2y, cv2.INTER_LINEAR , cv2.BORDER_CONSTANT, 0)

            frame_l_color = cv2.cvtColor(frame_l, cv2.COLOR_GRAY2BGR)
            frame_r_color = cv2.cvtColor(frame_r, cv2.COLOR_GRAY2BGR)

            for line in range(0, int(frame_l_color.shape[0] / 20)):
                frame_l_color = cv2.line(frame_l_color, (0, line*20), (frame_l_color.shape[1], line*20), (10, 255, 255))
                frame_r_color = cv2.line(frame_r_color, (0, line*20), (frame_r_color.shape[1], line*20), (10, 255, 255))

            stack = np.hstack((frame_l_color, frame_r_color))

            cv2.imshow("stack", stack)     

            disparity = stereo.compute(frame_l, frame_r)/16

            min = disparity.min()
            max = disparity.max()

            disparity = np.uint8(255 * (disparity - min) / (max - min))

            cv2.imshow("disparity", disparity)

            while True:
                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 32:
                    break

if __name__ == "__main__":
    stereo_cal = StereoImageCalibrate()

# flags = cv2.CALIB_FIX_INTRINSIC


# rectifyScale = 1
# rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roi_l, roi_r = cv2.stereoRectify(new_cam_mat_l, dist_l, new_cam_mat_r, dist_r, shape_l, rot, trans, rectifyScale,(0,0))


# stereo_map_l = cv2.initUndistortRectifyMap(new_cam_mat_l, dist_l, rect_l, proj_mat_l, shape_l, cv2.CV_16SC2)
# stereo_map_r = cv2.initUndistortRectifyMap(new_cam_mat_r, dist_r, rect_r, proj_mat_r, shape_r, cv2.CV_16SC2)

# print("Saving parameters!")
# cv_file = cv2.FileStorage('stereo_map.xml', cv2.FILE_STORAGE_WRITE)

# cv_file.write('stereo_map_l_x',stereo_map_l[0])
# cv_file.write('stereo_map_l_y',stereo_map_l[1])
# cv_file.write('stereo_map_r_x',stereo_map_r[0])
# cv_file.write('stereo_map_r_y',stereo_map_r[1])

# cv_file.release()

