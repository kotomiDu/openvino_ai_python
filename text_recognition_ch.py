"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import cv2
import numpy as np

from common import IEModel
import os
class Model(IEModel):
    """ Class that allows worknig with person detectpr models. """

    def __init__(self, model_path, device, ie_core, num_requests = 1):
        """Constructor"""

        super().__init__(model_path, device, ie_core, num_requests)
        _, _, h, w = self.input_size
        self.input_height = h
        self.input_width = w
        
        keys_path = os.path.join(os.path.dirname(model_path),'keys.txt')
        keys_file = open(keys_path, 'r', encoding ='utf-8')
        self.alphabet  = keys_file.readline().rstrip()


    def _prepare_frame(self, img):
        """Converts input image according model requirements"""
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        width,height = 120,32
        img = cv2.resize(img,(width,height))
        img_invert = np.zeros((height,width,1),np.uint8)
        for i in range(height):
            for j in range(width):
                grayPixel=img[i][j]
                img_invert[i][j]=255-grayPixel
        '''diffiduclt to check the text is black or white'''
        # cv2.imwrite("test1.jpg",img)
        # cv2.imwrite("test2.jpg",img_invert)
        # ret, img_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(img_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        # print(len(contours))
        # for contour in contours:
           # x, y, w, h = cv2.boundingRect(contour)
           # cv2.rectangle(img_invert, (x, y), (x + w, y + h), (0, 255, 255), 4)
        # cv2.imshow("show", img_bw)  
        # cv2.waitKey(0)  
        # boundary_array = np.concatenate((img[0, :], img[:, self.input_width - 1], img[self.input_height - 1, :], img[:, 0]), axis=0)
        # invert_boundary_array = np.concatenate((img_invert[0, :], img_invert[:, self.input_width - 1], img_invert[self.input_height - 1, :], img_invert[:, 0]), axis=0)
        # print(boundary_array,np.median(boundary_array),np.median(invert_boundary_array))
        
        img = np.array(img).astype(np.float32) / 255.0 - 0.5
        img_invert = np.array(img_invert).astype(np.float32) / 255.0 - 0.5

        img = img.reshape([1, 1,32, width])
        img_invert = img_invert.reshape([1, 1,32, width])
        return [img,img_invert]

    def _process_output(self,pred):
        """Converts network output to text"""
        characters = self.alphabet[:]
        characters = characters[1:] + u'卍'
        nclass = len(characters)
        char_list = []
        
        pred_text = pred.argmax(axis=2)[0]
        #print(pred.argmax(axis=2),pred_text)
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(characters[pred_text[i]])
        return u''.join(char_list)
    
    def __call__(self, frame):
        """Runs model on the specified input"""
        outs = []
        frames = self._prepare_frame(frame)
        for in_frame in frames:
            result = self.infer(in_frame)
            out = self._process_output(result)
            outs.append(out)   
        return outs
