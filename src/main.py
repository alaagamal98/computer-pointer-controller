from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandmarksDetection
from gaze_estimation import Model_GazeEstimation
from head_pose_estimation import Model_HeadPoseEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import argparse
import cv2
import os

def build_argparser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-fd", "--face_detection", required=True, type=str, help="Path to .xml file of Face Detection model.")

    parser.add_argument("-fl", "--face_landmark", required=True, type=str, help="Path to .xml file of Facial Landmark Detection model.")

    parser.add_argument("-hp", '--head_pose', required=True, type=str, help="Path to .xml file of Head Pose Estimation model.")

    parser.add_argument("-ge", '--gaze_estimation', required=True, type=str, help="Path to .xml file of Gaze Estimation model.")

    parser.add_argument("-i", "--input", required=True, type=str, help="Path to video file or enter cam for webcam.")

    parser.add_argument("-prob", "--prob_threshold", required=False, type=float, default=0.6, help="Specify confidence threshold which the value here in range(0, 1), default=0.6")

    parser.add_argument("-d", "--device", type=str, default="CPU", help="Provide the target device: CPU / GPU / VPU / FPGA, default= CPU")
    
    args = parser.parse_args()

    return args


def main(args):

    mouse_controller = MouseController('medium', 'fast')

    print("Model Loading..")

    face_detection = Model_FaceDetection(args.face_detection, args.device)
    face_landmark = Model_FacialLandmarksDetection(args.face_landmark, args.device)
    head_pose = Model_HeadPoseEstimation(args.head_pose, args.device)
    gaze_estimation = Model_GazeEstimation(args.gaze_estimation, args.device)
    
    print("Model loaded successfully")

    input_feeder  = InputFeeder(input_type='video', input_file=args.input)
    input_feeder.load_data()

    face_detection.load_model()
    head_pose.load_model()
    face_landmark.load_model()
    gaze_estimation.load_model()

    for frame in input_feeder.next_batch():
        try:
            frame.shape
        except Exception as err:
            break

        key = cv2.waitKey(60)

        face,face_coord = face_detection.predict(frame.copy(), args.prob_threshold)

        if type(face)==int:
            print("Unable to detect the face.")
            if key==27:
                break
            continue
        
        headPose = head_pose.predict(face.copy())
        
        left_eye, right_eye, eye_coord  = face_landmark.predict(face.copy())
        
        mouse_coord, gaze_vector = gaze_estimation.predict(left_eye, right_eye, headPose)
        
        cv2.imshow('video',frame)
        mouse_controller.move(mouse_coord[0], mouse_coord[1])


    input_feeder.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = build_argparser()
    main(args)
