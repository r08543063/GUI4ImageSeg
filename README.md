## GUI for Medical Image Segmentation

## Requirements
Some important required packages include:
* Python == 3.8
* pytorch==2.1.0
* PyQt==5.15.10
* torchvision==0.16.0
* torchaudio==2.1.0
* Basic python packages such as Numpy, OpenCV ......

# Usage
1. Data perparation
Dataset is arranged in the following format:
```
DATA/
|-- GUI4ImageSeg
|   |-- HEDseg_ui.py
|   |-- HEDseg_contain.py
|   |-- HEDseg_start.py
|   |-- source folder
|   |   |-- src_1.png
|   |   |-- src_2.png
|   |-- model weight folder
|   |   |-- weight_1.pth
|   |   |-- weight_2.pth
```
2. Usage
```
cd code
python HEDseg_start.py
```
![image](https://github.com/hohosoda/GUI-for-Medical-Image-Segmentation/blob/main/Usage.PNG)
