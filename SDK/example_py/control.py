import cv2
import torch
import numpy as np

from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from transformers import pipeline
from PIL import Image,ImageDraw
from matplotlib import pyplot as plt


def get_model():
    #model_path = "/path/to/ultralytics/yolov5"  # Update this path to the correct directory
    #yolo_model = torch.hub.load(model_path, 'custom', path=model_path, source='local')
    #model_path = '/yolov5s.pt'
    #yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',path = model_path, source = 'local')
    
    model_path = "./yolov5s.pt"  # Update this path to the correct directory
    yolo_model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')
    
    #yolo_model = torch.load('./yolov5s.pt')
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    return yolo_model, pipe

def image_process(image,yolo_model, pipe):
    image_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    depth = pipe(image_PIL)["depth"]
    depth_map= np.array(depth)
    results = yolo_model(image)
    boxes = results.xyxy[0].cpu().numpy()
    height, width = image.shape[:2]

    return boxes, depth_map, width, height

def is_object_partially_in_frame(x1_obj, y1_obj, x2_obj, y2_obj, x1_frame, y1_frame, x2_frame, y2_frame):
    # Check for overlapping areas
    if x1_obj < x2_frame and x2_obj > x1_frame and y1_obj < y2_frame and y2_obj > y1_frame:
        return True  # The object is partially inside the frame
    return False  # The object is completely outside the frame

def detect_navigation_action(objects, depth_map, image_width, image_height):
    """Determine navigation actions based on detection results and depth information"""

    # Define the ratio for the forward box and side boxes
    center_box_width_ratio = 0.35
    center_box_height_ratio = 0.5
    side_box_width_ratio = 0.25
    side_box_height_ratio = 0.7
    margin_ratio = 0.05  # Used to determine the margins

    # Calculate the dimensions and position of the forward box
    center_width = int(image_width * center_box_width_ratio)
    center_height = int(image_height * center_box_height_ratio)
    margin_x = int(image_width * margin_ratio)
    margin_y = int(image_height * margin_ratio)

    # Coordinates of the center box
    start_x_center = (image_width - center_width) // 2
    end_x_center = start_x_center + center_width
    start_y_center = (image_height - center_height) // 2
    end_y_center = start_y_center + center_height

    # Calculate the dimensions and position of the side boxes
    side_width = int(image_width * side_box_width_ratio)
    side_height = int(image_height * side_box_height_ratio)
    side_margin_y = (image_height - side_height) // 2

    # Coordinates of the left box
    start_x_left = margin_x
    end_x_left = start_x_left + side_width
    start_y_left = side_margin_y
    end_y_left = start_y_left + side_height

    # Coordinates of the right box
    start_x_right = image_width - margin_x - side_width
    end_x_right = start_x_right + side_width
    start_y_right = side_margin_y
    end_y_right = start_y_right + side_height

    # Initialize detection flags and depth information
    obstacle_in_center = False
    distance_in_center = 0.0
    obstacle_in_left = False
    obstacle_in_right = False
    avg_depth_left = 0
    avg_depth_right = 0

    # Check each detected object
    for box in objects:

        x1_obj, y1_obj, x2_obj, y2_obj, _, _ = box

        # Check if the object is within the forward box
        if is_object_partially_in_frame(x1_obj, y1_obj, x2_obj, y2_obj, start_x_center, start_y_center, end_x_center, end_y_center):
            obstacle_in_center = True
            # Get the maximum depth of the object within the forward box
            center_depth = depth_map[int(y1_obj):int(y2_obj), int(x1_obj):int(x2_obj)].mean()
            distance_in_center = max(distance_in_center, center_depth)

        # Check if the object is within the left box
        if is_object_partially_in_frame(x1_obj, y1_obj, x2_obj, y2_obj, start_x_left, start_y_left, end_x_left, end_y_left):
            obstacle_in_left = True
        avg_depth_left = depth_map[int(start_y_left):int(end_y_left), int(start_x_left):int(image_width//2)].mean()

        # Check if the object is within the right box
        if is_object_partially_in_frame(x1_obj, y1_obj, x2_obj, y2_obj, start_x_right, start_y_right, end_x_right, end_y_right):
            obstacle_in_right = True
        avg_depth_right = depth_map[int(start_y_right):int(end_y_right), int(image_width//2):int(end_x_right)].mean()

    
    # Decide action based on detection results
    danger_threshold = 200 # Danger distance threshold in meters
    avg_center_depth = depth_map[int(start_y_center):int(end_y_center), int(start_x_center):int(end_x_center)].mean()
    if distance_in_center > danger_threshold or avg_center_depth > danger_threshold:
        return 0 #0

    # Decide action based on detection results
    if obstacle_in_center and distance_in_center > 150:  # Distance threshold assumed to be 2.0 meters
        # Unable to move forward, consider moving left or right
        if not obstacle_in_left and obstacle_in_right:
            return 2 #2 left
        elif not obstacle_in_right and obstacle_in_left:
            return 3 #3right
        elif avg_depth_left > avg_depth_right:
            return 3
        else:
            return 2
    else:
        return 1 #1

def decision_model(image,yolo_model, pipe):
    boxes, depth_map, width, height= image_process(image,yolo_model, pipe)
    order=detect_navigation_action(boxes, depth_map, width, height)
    
    return order


if __name__ == "__main__":
    yolo_model, pipe=get_model()
    camera = cv2.VideoCapture(2)
    ret,image=camera.read()
    order= decision_model(image ,yolo_model, pipe )
    print(order)
