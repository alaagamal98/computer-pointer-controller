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
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def predict(self, image):
    
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:img_processed})
        coords = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32) 
        le_xmin=coords[0]-10
        le_ymin=coords[1]-10
        le_xmax=coords[0]+10
        le_ymax=coords[1]+10
        
        re_xmin=coords[2]-10
        re_ymin=coords[3]-10
        re_xmax=coords[2]+10
        re_ymax=coords[3]+10

        left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
        return left_eye, right_eye, eye_coords

    def check_model(self):  

        if self.device == "CPU":     
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)  
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                logging.error("[ERROR] Unsupported layers found: {}".format(unsupported_layers))
                sys.exit(1)

    def preprocess_input(self, image):

        image_processed = cv2.resize(image,(self.input_shape[3], self.input_shape[2]))
        image_processed = image_processed.transpose(2, 0, 1)
        image_processed = image_processed.reshape(1, *image_processed.shape)
        return image_processed

    def preprocess_output(self, outputs):

        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y) 