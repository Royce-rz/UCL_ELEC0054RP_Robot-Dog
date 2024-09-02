import sys
import time
import math
import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# Import the decision-making functions
from control import get_model, decision_model  # Replace with the actual name of your module

if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Running on CPU.")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0') 
# Ensure your model is on the GPU


# Initialize YOLO model and depth estimation pipeline
yolo_model, pipe = get_model()


def get_robot_command(order):
    cmd = sdk.HighCmd()
    if order == 0:  # Stop
        cmd.mode = 0
        cmd.velocity = [0, 0]
    elif order == 1:  # Move forward
        cmd.mode = 2
        cmd.gaitType = 2
        cmd.velocity = [0.4, 0]  # Move forward with velocity 0.4
    elif order == 2:  # Move left
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [0, -0.2]  # Move slightly forward
        cmd.yawSpeed = 1.0  # Turn left
    elif order == 3:  # Move right
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [0, 0.2]  # Move slightly forward
        cmd.yawSpeed = -1.0  # Turn right
    return cmd

if __name__ == '__main__':
    HIGHLEVEL = 0xee
    LOWLEVEL = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.12.245", 8082)

    state = sdk.HighState()
    cmd = sdk.HighCmd()
    udp.InitCmdData(cmd)

    # Load and process the image
    camera = cv2.VideoCapture(2)
    ret, image = camera.read()
    order = decision_model(image, yolo_model, pipe)
    cmd = get_robot_command(order)
    #
    motiontime = 0
    while True:
        time.sleep(0.001)
        motiontime += 1

        udp.Recv()
        udp.GetRecv(state)

        # Update the command if necessary
        if motiontime % 1000 == 0:  # Adjust this frequency as needed
            ret, image = camera.read()
            order = decision_model(image, yolo_model, pipe)
            cmd = get_robot_command(order)
            print(order)
        #udp.SetSend(cmd)
  
        #udp.Send()
    camera.release()
    cv2.destroyAllWindows()
