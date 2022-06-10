from numpy import real
import rclpy
# import the ROS2 python libraries
from rclpy.node import Node
# import the LaserScan module from sensor_msgs interface
from sensor_msgs.msg import Image
#import our custom bounding box message
from yolo_msgs.msg import BoundingBox
#convert the image and into opencv image
from cv_bridge import CvBridge
from rclpy.qos import ReliabilityPolicy, QoSProfile

#dependencies for running our subscriber and publisher
import os
import cv2
import time
from pathlib import Path
import numpy as np

#yolov5 dependencies
import torch
import torch.backends.cudnn as cudnn

#imports from yolo repository
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from PIL import Image as PILImage
import torchvision.transforms as T


bridge = CvBridge()

class CarDetector(Node):

    def __init__(self):
        
        #constructor
        super().__init__('car_detector')

        #create the subscriber
        self.subscriber = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        #prevent unused variable warning
        self.subscriber

        #create the publisher
        self.publisher_ = self.create_publisher(BoundingBox, '/yolov5_boxes', 10)
        self.image_publisher = self.create_publisher(Image, '/bounding_images', 10)


        #create a bridge to convert messages into images
        self.bridge = CvBridge()
        self.image_count = 0
        self.image_msg = Image()

        #published messages
        self.detection = False
        self.confidence = 0.0
        self.x = 0.0
        self.y = 0.0
        self.width = 0.0
        self.height = 0.0
        self.detections = []
        

        # define the timer period for 0.5 seconds
        self.timer_period = 0.5
        self.timer = self.create_timer(self.timer_period, self.publish_box)

        #parameters for the yolov5 model

        self.weights = '/home/projects/ros2_ws/src/dsc-178/yolov5/runs/train/car_det/weights/best.pt' #directory of the weights
        self.data = '/home/projects/ros2_ws/yolov5/data.yaml' #dataset.yaml path
        self.device = select_device('cpu') #cuda device, 0 or 0,1,2,3 or cpu
        self.conf_thres = .25 #confidence threshold
        self.iou_thres = .15 # NMS IOU threshold
        self.classes = None #filter by class (0, 0, 2, 3)
        self.agnostic_nms = False # class-agnostic NMS
        self.max_det = 1000 #max detections per image
        self.line_thickness = 3

        #initialize our model
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, fp16=False)

        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.size = check_img_size(((1280, 736)), s=self.stride)  # check image size

        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.size))  # warmup



    def image_callback(self, msg):

        #convert the message into an OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img = cv2.resize(img, (640, 640))
        im0 = img

        img = img[np.newaxis, :, :, :] 
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        #flip our image into the right format for pytorch



        # img = PILImage.fromarray(img)
        # #convert to PIL image

        # process = T.Compose([
        #     T.ToTensor()
        # ])
        # #generate our compose object to change our image into a tensor

        # im = process(img)
        # #process our image

        # Stack
        # img = np.stack(img, 0)
        

        #process the image into tensor
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        pred = self.model(im, augment = False, visualize = False)
        #make our prediction
        self.get_logger().info('Publishing (before NMS): "%s"' % str(len(list(pred))))

        #NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        det = list(pred)
        self.detections = []
        self.get_logger().info('Publishing: "%s"' % str(len(det)))
        for det in list(pred):
            if len(det) != 0:
                #get our predictions
                self.x, self.y, self.width, self.height, self.confidence, img_class = det[0]
                #save them to publish

                if self.confidence > .5:
                    self.detection = True
                else:
                    self.detection = False
                #if we see car, true, else false

                self.detections.append([self.x, self.y, self.width, self.height, self.confidence, self.detection, img_class])

            else:
                self.confidence = 0.0
                self.x = 0.0
                self.y = 0.0
                self.width = 0.0
                self.height = 0.0
                img_class = ''
                self.detections.append([self.x, self.y, self.width, self.height, self.confidence, False, img_class])

            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            #add a bounding box using utils' annotator. line_thickness in pixels, self.names is from model
            annotator.box_label([self.x, self.y, self.width, self.height], 'Car, ' + str(np.array(self.confidence)), color = (255, 0, 0)) #we removed multiclass detection for now

        output_img = cv2.resize(im0, (1280,736))
        #resize to original bounding box

        self.image_msg = self.bridge.cv2_to_imgmsg(output_img)

       # cur_dir = os.getcwd()
       # success = cv2.imwrite(cur_dir + '/test_images/image' + str(self.image_count) + '.jpg', output_img)
       # self.image_count += 1
        #override this later, this is going to be published at a later date
    
    def publish_box(self):
        msg = BoundingBox()

        # msg.x = float(self.x)
        # msg.y = float(self.y)
        # msg.width = float(self.width)
        # msg.height = float(self.height)
        # msg.probability = float(self.confidence)
        # msg.detection = self.detection
        #generate our message

        # Publish the message to the topic
        for det in self.detections:
            msg.x = float(det[0])
            msg.y = float(det[1])
            msg.width = float(det[2])
            msg.height = float(det[3])
            msg.probability = float(det[4])
            msg.detection = det[5]
            self.publisher_.publish(msg)

        self.image_publisher.publish(self.image_msg)
        # Display the message on the console
        self.get_logger().info('Publishing: "%s"' % msg)
    
def main(args = None):
    #initialize the ROS communication
    rclpy.init(args = args)
    #declare the node constructor
    car_detector = CarDetector()
    #puase the program execution, waits for ctrl-c
    rclpy.spin(car_detector)
    #destroy the node
    car_detector.destroy_node()
    #shutdown ros communication
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
