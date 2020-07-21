import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import sys
import logging

class Model_FacialLandmarksDetection:

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
        self.out_shape = None

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
        self.out_name = next(iter(self.network.outputs))
        self.out_shape = self.network.outputs[self.out_name].shape

    def predict(self, image):
    
        processed_image = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.in_name:processed_image})
        coord = self.preprocess_output(outputs)
        height=image.shape[0]
        width=image.shape[1]

        coord = coord* np.array([width, height, width, height])
        coord = coord.astype(np.int32) 

        left_xmin=coord[0]-10
        left_ymin=coord[1]-10
        left_xmax=coord[0]+10
        left_ymax=coord[1]+10
        
        right_xmin=coord[2]-10
        right_ymin=coord[3]-10
        right_xmax=coord[2]+10
        right_ymax=coord[3]+10

        left_eye =  image[left_ymin:left_ymax, left_xmin:left_xmax]
        right_eye = image[right_ymin:right_ymax, right_xmin:right_xmax]
        eye_coord = [[left_xmin,left_ymin,left_xmax,left_ymax], [right_xmin,right_ymin,right_xmax,right_ymax]]
        return left_eye, right_eye, eye_coord

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

        outs = outputs[self.out_name][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y) 