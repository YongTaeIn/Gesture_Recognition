# __Gesture_Recognition.__
Framework : Pytorch, MediaPipe

This repository contains all the code(model, make_dataset etc) so you can customize it.

***

## __How to do environment setup.__
1. conda create -n Gesture
2. pip install -r requirements.txt

***

## __Structure of directory.__

```bash
├── development_files_for_reference
│   ├── CustomDataset.ipynb
│   ├── model.ipynb
│   └── validation
│ 
├── images
│   └── model.png
│ 
├── main_data
│   ├── data_gesture_Left.csv
│   ├── data_gesture_Right.csv
│   ├── data_gesture_Turn Anticlockwise.csv
│   └── data_gesture_Turn Clockwise.csv
│ 
├── main_files
│   ├── CustomDataset.py
│   ├── make_dataset.ipynb
│   └── model.py
│ 
├── main_models
│   ├── model_dict().pt
│   └── model.pt
│ 
├── test.ipynb
└── train.ipynb
``` 


***

## __Explanation about folder.__
development_files_for_reference :  Miscellaneous files used during development (feel free to delete)

main_data : Path where collected data is stored: We collected left, right clockwise and counterclockwise data.

main_files :
- CustomDataset.py : Implement customdataloader as a class using pytorch.
- make_dataset.ipynb : Use it when collecting data.
-
<span style="color:yellow">You must change class_num when you gather other dataset.</span>


- model.py : CNN-LSTM model was implemented as class type.

main_models : model save path 

test.ipynb : Code used when testing the model

train.ipynb: Code used when training the model

***

## __Model architecture.__
<img src="/images/model.png" width="1000px" height="600px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

***
## __Precautions for use.__
1. mediapipe need python version 3.8 - 3.11 , Beware of version conflicts! (If you used requirement.txt, no need to worry)
2. If you use a public server rather than a local computer, an error will occur in your code because there is no camera.
3. So i did data collect in local computer with camera attached and test in local computer. Only Traning process is used in School Server.


***
## __ETC__
1. Early stopping was not applied here.
