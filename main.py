import logging as log
import sys
import time
import json
from os.path import exists
from argparse import ArgumentParser, SUPPRESS
import glob
import os

import cv2
import numpy as np
from common import load_ie_core
from text_detection import Model as DetectionModel

from text_recognition_ch import Model as chRecognitionModel
from text_recognition_eng import Model as engRecognitionModel

def build_argparser():
    """ Returns argument parser. """
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                          help='Required. Path to an .xml file with a general model.',
                          required=True, type=str)
    args.add_argument('-m_d', '--detection_model',
                          help='Required. Path to an .xml file with a trained person detector model.',
                          required=False, type=str)
    args.add_argument('-i', '--input',
                          help='Required. Path to a video file or image folder.',
                          required=False, type=str)
    args.add_argument('-d', '--device',
                          help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL '
                               'or MYRIAD. The demo will look for a suitable plugin for device '
                               'specified (by default, it is CPU).',
                          default='CPU', type=str)
    args.add_argument("-l", "--cpu_extension",
                          help="Optional. Required for CPU custom layers. Absolute path to "
                               "a shared library with the kernels implementations.", type=str,
                          default=None)
    return parser




def test_text_detection():  
    #ie_core = load_ie_core(args.device, args.cpu_extension)
    text_detection_model = DetectionModel(args.model, args.device, ie_core)
    if os.path.isdir(args.input):
        imglist = glob.glob(args.input)
    else:
        imglist = [args.input]
    
    for idx,imfn in enumerate(imglist):
        oriim = cv2.imread(imfn)
        bboxes  = text_detection_model(oriim)
        if len(bboxes[0]) > 0:    
            for i , box in enumerate(bboxes[0]):
                pts = np.array(box).reshape((-1,1,2))
                cv2.polylines(oriim,[pts],True,(0,255,255),2)
        cv2.imshow("show",oriim)
        cv2.waitKey(0) 
        

def test_text_recogntion():  
    if 'eng' in args.model:
        text_recognition_model = engRecognitionModel(args.model, args.device, ie_core)
    elif 'ch' in args.model:
        text_recognition_model = chRecognitionModel(args.model, args.device, ie_core)
    if os.path.isdir(args.input):
        imglist = glob.glob(args.input)
    else:
        imglist = [args.input]
    for imgpath in imglist:
        oriim = cv2.imread(imgpath)
        print(imgpath)
        out  = text_recognition_model(oriim)
        #print(out)
        
        def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 255), textSize=80):
            from PIL import Image,ImageDraw,ImageFont 
            if (isinstance(img, np.ndarray)):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
           
            fontStyle = ImageFont.truetype(
                "font/simsun.ttc", textSize, encoding="utf-8")
           
            draw.text((left, top), text, textColor, font=fontStyle)
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        h,w,_ = oriim.shape
        padding_image = np.zeros((h*2,w,3), np.uint8)
        if type(out).__name__ != 'list':
            out = [out]
        for i,res in enumerate(out):
            padding_image = cv2ImgAddText(padding_image,res, 0, i*20)
        res_im = np.concatenate((padding_image, oriim), axis=0)
        cv2.imshow("show",res_im)
        cv2.waitKey(0) 


if __name__ == '__main__':
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    ie_core = load_ie_core(args.device, args.cpu_extension)
    #sys.exit(test_ov() or 0)            
    if  "detect" in args.model:
        test_text_detection()
    if "recog" in args.model():
        test_text_recogntion()