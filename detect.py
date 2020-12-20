from __future__ import division

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import cv2

import os
import random

from darknet import Darknet
from util import *

def plot_rectangle(out, load_imgs):
    c1 = tuple(out[1:3].int())
    c2 = tuple(out[3:5].int())
    color = random.choice(colors)
    img = load_imgs[int(out[0])]

    cv2.rectangle(img, c1, c2, color, 1)
    label = classes[int(out[-1])]
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    c2 = c1[0] + t_size[0] +3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)

    return img

CUDA = torch.cuda.is_available()
args = args_parse()
model = Darknet('cfg/yolov3.cfg')
model.load_weights('./yolov3.weights')
if CUDA:
    model = model.cuda()

model.eval()

num_classes = 80
classes = load_class('data/coco.names')

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

images = args.images
try:
    im_list = [os.path.join(os.path.realpath('.'), images, img)  for img in os.listdir(images)]
except NotADirectoryError:
    im_list = [os.path.join(os.path.realpath('.'), images)]
except FileNotFoundError:
    print('No file or diroctory named {}'.format(images))

load_ims = [cv2.imread(img) for img in im_list]

inp_dim = args.input_size
prep_ims = list(map(prep_img, load_ims, [inp_dim for i in range(len(im_list))]))
print(prep_ims[0][0,0,0,:10])
pdb

im_dim_list = [(img.shape[1], img.shape[0]) for img in load_ims]
im_dim_list = torch.FloatTensor(im_dim_list)

if CUDA:
    im_dim_list = im_dim_list.cuda()

batch_size = int(args.batch_size)
leftover = 0
if len(im_list) % batch_size != 0:
    leftover = 1

if batch_size > 1:
    num_batchs = len(im_list) // batch_size + leftover
    im_batchs = [torch.cat((prep_ims[i*batch_size : min((i+1)*batch_size, len(prep_ims))]))  for i in range(num_batchs)]
else:
    im_batchs = prep_ims

write = 0
for i, batch in enumerate(im_batchs):
    if CUDA:
        batch = batch.cuda()

    prediction = model(batch)
    prediction = write_results(prediction, confidence=args.confidence, nms_conf=args.nms_conf)
    print(prediction)
    pdb

    if type(prediction) == int:
        print('Nothing detected!')
        continue

    prediction[:, 0] += i * batch_size

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for idx, image in enumerate(im_list[i*batch_size : min((i+1)*batch_size,len(im_list))]):
        im_id = i * batch_size + idx
        objs = [classes[int(obj[-1])] for obj in output if int(obj[0]) == im_id]
        print('Objects detected:', ','.join(objs))

    if CUDA:
        torch.cuda.synchronize()

try:
    output
except:
    print('No detections!')

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scale_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1,1)
output[:,[1,3]] -= (inp_dim - scale_factor * im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scale_factor * im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scale_factor

for i in range(output.shape[0]):
    output[i,[1,3]] = torch.clamp(output[i,[1,3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

colors = pkl.load(open('pallete', 'rb'))

list(map(lambda x: plot_rectangle(x, load_ims), output))

print(im_list)
save_names = pd.Series(im_list).apply(lambda x: "{}/det_{}".format(args.save_path, x.split('\\')[-1]))
print(save_names)
list(map(cv2.imwrite, save_names, load_ims))




