import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt


class PoseEstimator:
    def __init__(self, pose_weights="yolov7-w6-pose.pt", source="football1.mp4", device='cpu',
                line_thickness=3, hide_labels=False, hide_conf=True):
        
        self.pose_weights = pose_weights
        self.source = source
        self.device = device
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.model = None
        self.names = None
        self.cap = None

    def __run(self):
        frame_count = 0
        total_fps = 0
        time_list = []
        fps_list = []

        self.device = select_device(self.device)
        half = self.device.type != 'cpu'

        self.model = attempt_load(self.pose_weights, map_location=self.device)
        _ = self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.source.isnumeric():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print('Error while trying to read video. Please check path again')
            raise SystemExit()

        else:
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))
            while self.cap.isOpened:

                ret, frame = self.cap.read()

                if ret:
                    orig_image = frame
                    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                    image_ = image.copy()
                    image = transforms.ToTensor()(image)
                    image = torch.tensor(np.array([image.numpy()]))

                    image = image.to(self.device)
                    image = image.float()
                    start_time = time.time()

                    with torch.no_grad():
                        output_data, _ = self.model(image)

                    output_data = non_max_suppression_kpt(output_data, 0.25, 0.65,
                                                          nc=self.model.yaml['nc'],
                                                          nkpt=self.model.yaml['nkpt'], kpt_label=True)

                    output = output_to_keypoint(output_data)

                    im0 = image[0].permute(1, 2, 0) * 255
                    im0 = im0.cpu().numpy().astype(np.uint8)

                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                    for i, pose in enumerate(output_data):
                        if len(output_data):
                            for c in pose[:, 5].unique():
                                n = (pose[:, 5] == c).sum()
                                print("No of Objects in Current Frame : {}".format(n))

                            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                                c = int(cls)
                                kpts = pose[det_index, 6:]
                                label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                                plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                                 line_thickness=self.line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                                 orig_shape=im0.shape[:2])

                    end_time = time.time()
                    fps = 1 / (end_time - start_time)
                    total_fps += fps
                    frame_count += 1
                    avg_fps = total_fps / frame_count
                    print(f"FPS: {avg_fps:.3f}")
                    fps_list.append(total_fps)
                    time_list.append(end_time - start_time)
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)
                    
                else:
                    break

            self.cap.release()




    def main(self,opt):
        pose_estimator = PoseEstimator(pose_weights=opt.poseweights, source=opt.source, device=opt.device,
                                    line_thickness=opt.line_thickness,
                                    hide_labels=opt.hide_labels, hide_conf=opt.hide_conf)
        pose_estimator.__run()



