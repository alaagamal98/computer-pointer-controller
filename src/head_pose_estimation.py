import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import sys
import logging

class Model_HeadPoseEstimation:

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
        
        self.in_name = next(iter(self.network.inputs))
        self.in_shape = self.network.inputs[self.in_name].shape
        self.out_name = [i for i in self.network.outputs.keys()]

    def predict(self, image):
    
        processed_image = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.in_name:processed_image})
        final = self.preprocess_output(outputs)
        return final

    def check_model(self):  

        if self.device == "CPU":     
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)  
            notsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

            if len(notsupported_layers) != 0:
                logging.error("[ERROR] Unsupported layers found: {}".format(notsupported_layers))
                sys.exit(1)

    def preprocess_input(self, image):

        image_processed = cv2.resize(image,(self.in_shape[3], self.in_shape[2]))
        image_processed = image_processed.transpose(2, 0, 1)
        image_processed = image_processed.reshape(1, *image_processed.shape)
        return image_processed

    def preprocess_output(self, outputs):

        preprocessed_outputs = []
        preprocessed_outputs.append(outputs['angle_y_fc'].tolist()[0][0])
        preprocessed_outputs.append(outputs['angle_p_fc'].tolist()[0][0])
        preprocessed_outputs.append(outputs['angle_r_fc'].tolist()[0][0])
        return preprocessed_outputs