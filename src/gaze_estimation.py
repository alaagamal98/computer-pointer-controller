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
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
  
        model_structure = self.model_name
        model_weights = self.model_name.split('.')[0]+'.bin'

        self.plugin = IECore()
      
        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions,self.device)

        self.network = IENetwork(model=model_structure, weights=model_weights)

        self.check_model()

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, hpa):
    
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector

    def check_model(self):  

        if self.device == "CPU":     
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)  
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                logging.error("[ERROR] Unsupported layers found: {}".format(unsupported_layers))
                sys.exit(1)

    def preprocess_input(self, left_eye,right_eye):

        image_processed_right = cv2.resize(right_eye,(self.input_shape[3], self.input_shape[2]))
        image_processed_right = image_processed_right.transpose(2, 0, 1)
        image_processed_right = image_processed_right.reshape(1, *image_processed_right.shape)

        image_processed_left = cv2.resize(left_eye,(self.input_shape[3], self.input_shape[2]))
        image_processed_left = image_processed_left.transpose(2, 0, 1)
        image_processed_left = image_processed_left.reshape(1, *image_processed_left.shape)
        return image_processed_right, image_processed_left

    def preprocess_output(self, outputs,hpa):

        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        rollValue = hpa[2]
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector