import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import sys
import logging
import math

class Model_GazeEstimation:

    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.in_name = None
        self.in_shape = None
        self.out_name = None
        
    def load_model(self):
  
        model_structure = self.model_name
        model_weights = self.model_name.split('.')[0]+'.bin'

        self.plugin = IECore()
      
        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions,self.device)

        self.network = IENetwork(model=model_structure, weights=model_weights)

        self.check_model()

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.in_name = [i for i in self.network.inputs.keys()]
        self.in_shape = self.network.inputs[self.in_name[1]].shape
        self.out_name = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, head_angle):
    
        left_image_processed, right_image_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':head_angle, 'left_eye_image':left_image_processed, 'right_eye_image':right_image_processed})
        mouse_coords, gaze = self.preprocess_output(outputs,head_angle)

        return mouse_coords, gaze

    def check_model(self):  

        if self.device == "CPU":     
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)  
            notsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

            if len(notsupported_layers) != 0:
                logging.error("[ERROR] Unsupported layers found: {}".format(notsupported_layers))
                sys.exit(1)

    def preprocess_input(self, left_eye,right_eye):

        image_processed_right = cv2.resize(right_eye,(self.in_shape[3], self.in_shape[2]))
        image_processed_right = image_processed_right.transpose(2, 0, 1)
        image_processed_right = image_processed_right.reshape(1, *image_processed_right.shape)

        image_processed_left = cv2.resize(left_eye,(self.in_shape[3], self.in_shape[2]))
        image_processed_left = image_processed_left.transpose(2, 0, 1)
        image_processed_left = image_processed_left.reshape(1, *image_processed_left.shape)

        return image_processed_right, image_processed_left

    def preprocess_output(self, outputs,head_angle):

        gaze = outputs[self.out_name[0]].tolist()[0]
        roll_value = head_angle[2]
        cos_value = math.cos(roll_value * math.pi / 180.0)
        sin_value = math.sin(roll_value * math.pi / 180.0)
        
        x = gaze[0] * cos_value + gaze[1] * sin_value
        y = -gaze[0] *  sin_value+ gaze[1] * cos_value
        return (x,y), gaze