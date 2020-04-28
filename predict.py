import os
import sys
import cv2
import json
import numpy as np
import time
import torch
import torch.nn as nn 
from lib.models.resnet import resnet101_normal
from torchvision import transforms, utils, models
from yolov3.detper import DetectPerson
from PIL import Image

classes = ['no', 'stick', 'gun', 'knife', 'dao']
class Weapon_detect(object):
    def __init__(self):
        self.gpu = 0
        self.criterion = nn.CrossEntropyLoss()
        model = resnet101_normal()
        self.model = model.to(self.gpu)
        self.model.load_state_dict(
            torch.load('./save_models/weapon_resnet_99'))
    
    def detect(self,img):
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        image = image.resize((224,224))
        image = np.array(image)
        img_tensor = preprocess(image).unsqueeze(0)
        if torch.cuda.is_available():
            inputs = img_tensor.cuda(self.gpu)
        output = self.model(inputs)
        pred = output.data.max(1, keepdim=True)[1]
        return pred

dt = DetectPerson(confidence=0.9)
wd = Weapon_detect()
def detect_person(im0, H, W):
    
    '''
    0: 人
    '''
    det_cls = 0
    t = time.time()
    img = None
    #检测行人
    det = dt.detection(im0)
    det_list = []
    print('====================t1:',time.time()-t)
    if det is not None:
        for *xyxy, conf, _, cls in det:
            if cls == det_cls:
                label = '%s ' % (dt.classes[int(cls)])
                #左上右下坐标
                x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                w = x_max - x_min
                h = y_max - y_min
                x_w = w // 8 
                y_h = h // 8
                if x_min - x_w < 0:
                    x_min = 0
                else:
                    x_min -= x_w
                if x_max + x_w > W:
                    x_max = W
                else:
                    x_max += x_w
                if y_min - y_h < 0:
                    y_min = 0
                else:
                    y_min -= y_h
                if y_max + y_h > H:
                    y_max = H
                else:
                    y_max += y_h
                
                c1, c2 = (x_min, y_min), (x_max, y_max)
                # print(c1, c2)
                cv2.rectangle(im0,c1,c2,(255,0,0),1)
                #取出目标区域
                detect_im0 = im0[c1[1]:c2[1],c1[0]:c2[0],:]
                cls2 = wd.detect(detect_im0)

                font = cv2.FONT_HERSHEY_SIMPLEX

                img = cv2.putText(im0, classes[cls2.item()], c2, font, 1.2, (255, 255, 255), 2)
                

                
    return img

preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize
])





class ReadVideo(object):
    def __init__(self):
        pass
        
    def handle(self, video_path):
        
        video_path_trim = video_path.split('/')[-1]
        video_name = video_path_trim.split('.')[0] + '.mp4'
        video_path_finish = 'video_finish/'
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        w = frame.shape[1]
        h = frame.shape[0]
        num = 0
        videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','P','4','V'), 24, (w,h))
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            
            if ret:
                if num % 2 == 0:
                    st = time.time()
                    img = detect_person(frame, h, w)
                    if img is None:
                        img = frame
                    et = time.time()
                    cv2.imwrite('./test.jpg', img)
                    time.sleep(1)
                    print('--.--.--.--.--.--总共花时 Algorithm cost %fs--.--.--.--.--.--' %(et-st))
                    # cv2.imwrite('/home/Project/structured/1.jpg',image)
                    videoWriter.write(img)
            else:
                break
            num += 1
            
        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()





if __name__ == '__main__':
    rd = ReadVideo()
    rd.handle('./video/2-1.mp4')