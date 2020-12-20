import torch
import cv2
import numpy as np
import argparse

def prediction_transfer(prediction, anchors, inp_dim, num_classes):
    #torch.Size([1, 255, 13, 13]) -> torch.size([k, 8])
    batch_size = prediction.size(0)
    grid_size = prediction.size(2)
    stride = inp_dim // grid_size

    num_anchors = len(anchors)
    num_attrs = 5 + num_classes

    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    prediction = prediction.view(batch_size, num_anchors * num_attrs, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, num_attrs)

    #torch.Size([1, 507, 85])
    prediction[:,:,:2] = torch.sigmoid(prediction[:,:,:2])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    x = np.arange(grid_size)
    a, b = np.meshgrid(x,x)
    a = torch.FloatTensor(a).view(-1,1)
    b = torch.FloatTensor(b).view(-1,1)
    x_y_offset = torch.cat((a,b), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors).repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    prediction[:,:,:4] *= stride
    return prediction

def compute_ious(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_left = torch.max(box1_x1, box2_x1)
    inter_top = torch.max(box1_y1, box2_y1)
    inter_right = torch.min(box1_x2, box2_x2)
    inter_down = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp((inter_right - inter_left), min=0) * torch.clamp((inter_down - inter_top), min=0)

    box1_area = (box1_y2 - box1_y1) * (box1_x2 - box1_x1)
    box2_area = (box2_y2 - box2_y1) * (box2_x2 - box2_x1)

    ious = inter_area / (box1_area + box2_area - inter_area)

    return ious

def write_results(prediction, confidence, nms_conf):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.shape[0]
    write = 0
    for ind in range(batch_size):
        prediction_ind = prediction[ind]

        max_score, max_index = torch.max(prediction_ind[:,5:], 1)

        max_score = max_score.unsqueeze(1)
        max_index = max_index.unsqueeze(1)

        seq = prediction_ind[:,:5], max_score, max_index
        prediction_ind = torch.cat(seq, 1)

        non_zero = torch.nonzero(prediction_ind[:, 4]).long()
        prediction_ind = prediction_ind[non_zero, :].view(-1,7)
        #15, 7

        classes = torch.unique(prediction_ind[:, -1].detach())

        for cls in classes:
            cls_mask = (prediction_ind[:, -1] == cls).float().unsqueeze(1)
            prediction_cls = prediction_ind * cls_mask
            non_zero = torch.nonzero(prediction_cls[:,-1])
            prediction_cls = prediction_cls[non_zero, :].view(-1, 7)

            cls_sort = torch.sort(prediction_cls[:,4], descending=True)[1]
            prediction_cls = prediction_cls[cls_sort]

            for idx in range(prediction_cls.shape[0]):
                try:
                    ious = compute_ious(prediction_cls[idx].unsqueeze(0), prediction_cls[idx + 1:])
                except:
                    break

                iou_conf = (ious < nms_conf).float().unsqueeze(1)
                prediction_cls[idx + 1:] *= iou_conf
                non_zero = torch.nonzero(prediction_cls[:, 0])
                prediction_cls = prediction_cls[non_zero].view(-1, 7)

            batch_ind = prediction_cls.new(prediction_cls.size(0), 1).fill_(ind)
            seq = (batch_ind, prediction_cls)

            if not write:
                outputs = torch.cat(seq, 1)
                write = 1
            else:
                out = torch.cat(seq, 1)
                outputs = torch.cat((outputs, out))

    try:
        return outputs
    except:
        return 0

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416,416))
    img = img[:,:,::-1].transpose((2,0,1))# .copy()
    img = img[np.newaxis, ...] / 255.0
    img = torch.from_numpy(img).float()

    return img

def args_parse():
    parser = argparse.ArgumentParser(description='Yolov3 Arguments')

    parser.add_argument('--images', dest='images', help='image path', default='./img')
    parser.add_argument('--save_path', help='save path', default='./results')
    parser.add_argument('--cfg', help='cfg_path', default='./cfg/yolov3.cfg')
    parser.add_argument('--weights', help='weights path', default='./yolov3.weights')
    parser.add_argument('--input_size', help='input size', default=416)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--confidence', help='confidence to filter detections', default=0.5)
    parser.add_argument('--nms_conf', help='threshhold fpr nms', default=0.4)

    return parser.parse_args()

def load_class(class_file):
    fp = open(class_file, 'r')
    classes = fp.readlines()
    classes = [line.rstrip('\n') for line in classes]
    return classes

def letter_box(img, inp_dim):
    h, w = img.shape[1], img.shape[0]

    scale = min(inp_dim/h, inp_dim/w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mask = np.full((inp_dim, inp_dim, 3), 128)
    mask[(inp_dim - new_h) // 2: ((inp_dim - new_h) // 2 + new_h), (inp_dim - new_w) // 2: ((inp_dim - new_w) // 2 + new_w), :] = img

    return mask

def prep_img(img, inp_dim):
    img = letter_box(img, inp_dim)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img






