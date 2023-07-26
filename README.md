# yolov7-pose-estimation

### Features

- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported


### Steps to run Code
original repo--> git clone https://github.com/RizwanMunawar/yolov7-pose-estimation.git

my repo with oop--> git clone https://github.com/Fatih-Haslak/Yolov7-pose-estimation-oop-based.git (use this for oop)
- Goto the cloned folder.
```
cd yolov7-pose-estimation
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv psestenv
source psestenv/bin/activate

### For Window Users
python3 -m venv psestenv
cd psestenv
cd Scripts
activate
cd ..
cd ..
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.

```
pip install -r requirements.txt
```

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}


- Run the code with mentioned command below.
```
python main.py

#if you want to change source file
python main.py --source "your custom video.mp4"

#For CPU
python main.py --source "your custom video.mp4" --device cpu

#For GPU
python main.py --source "your custom video.mp4" --device 0

#For WebCam
python main.py --source 0 

#For External Camera
python main.py --source 1 
```


#### RESULTS

<table>
  <tr>
    <td>Football Match Pose-Estimation</td>
     <td>Cricket Match Pose-Estimation</td>
     <td>FPS and Time Comparision</td>
     <td>Live Stream Pose-Estimation</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185089411-3f9ae391-ec23-4ca2-aba0-abf3c9991050.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185228806-4ba62e7a-12ef-4965-a44a-6b5ba9a3bf28.png" width=640 height=180></td>
  </tr>
 </table>

#### References
- https://github.com/WongKinYiu/yolov7
- https://github.com/augmentedstartups/yolov7
- https://github.com/augmentedstartups
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
- https://github.com/ultralytics/yolov5

#### Rizwan Munawar Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
