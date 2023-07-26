import argparse
from oop_pose_estimate import PoseEstimator

class Pose:


    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')
        parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        opt = parser.parse_args()
        return opt

if __name__ == "__main__":
    pose=PoseEstimator() # Yolov7 pose estimator class
    start=Pose() # Arg main class
    opt=start.parse_opt() # get args
    pose.main(opt) #run yolov7 pose estimator

