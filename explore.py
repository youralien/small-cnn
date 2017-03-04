#!/usr/bin/python

import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import Image, cv2
import numpy as np
import datetime

def image_from_webcam():
    cap = cv2.VideoCapture(0) # says we capture an image from a webcam
    _,cv2_im = cap.read()
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im

imsize = 277 # desired size of the output image

loader = transforms.Compose([
            transforms.Scale(imsize), # scale imported image
            transforms.ToTensor()]) # transform it into a torch tensor

def image_loader(image=None):
    if image is None:
	image = image_from_webcam() 
    image = Variable(loader(image))
    image = image.unsqueeze(0) # fake batch dimension required to fit network's input dimensions
    return image

squeezenet = models.squeezenet1_0(pretrained=True)

timetracker = np.zeros(100) 
for i in range(100):
    a = datetime.datetime.now()
    image = image_loader()
    res = squeezenet.forward(image)
    b = datetime.datetime.now()
    c = b - a
    timetracker[i] = c.microseconds

print "Average Time (us): {}".format(timetracker.mean())
