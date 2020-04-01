# OpenVINO AI
Intel OpenVINO AI usage

## Enviroment
* python3 **!! Use anaconda to manage python environment is strongly recommended**
* openvino 2020R1 version.
  please refer to [OpenVINO@intel windows installation](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)
  **!! Need to run setupvars.bat after installation**
* pip install opencv-python

### if you manage enviroment with Anaconda, here is some installation reference.
 <img src="./doc/pics/step1.png" width = "400" height = "200" alt="open command" align=left />
 <img src="./doc/pics/step2.png" width = "400" height = "200" alt="set environment"  />


## Run app
`python main.py -m model\text_detection\text-detection-0004.xml -i input\text_detection.png`
 <img src="./doc/pics/detection_result.png" width = "400" height = "200"  align=left />

`python main.py -m model\text_recognition_eng\text-recognition-0012.xml -i input\eng_recognition.png`
 <img src="./doc/pics/ch_result.jpg" width = "400" height = "200"  align=left />
 
`python main.py -m model\text_recognition_ch\text-recognition.xml -i input\ch_recognition.jpg`
 <img src="./doc/pics/eng_result.jpg" width = "400" height = "200"  align=left />
