import torch
import torch.nn as nn
import numpy as np

from util import *

def cfg_parse(cfg_file):
    blocks = []
    with open(cfg_file, 'r') as fp:
        lines = fp.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = [line for line in lines if len(line) > 0 and line[0] != '#']

    block = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1]
        else:
            block[line.split('=')[0].rstrip()] = line.split('=')[1].lstrip()

    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    outputs = []

    modules = blocks[1:]
    for i, block in enumerate(modules):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            kernel_size = int(block['size'])
            filters = int(block['filters'])
            stride = int(block['stride'])
            pad = int(block['pad'])
            active = block['activation']

            try:
                bn = int(block['batch_normalize'])
                bias = False
            except:
                bn = 0
                bias = True

            if pad == 1:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
            module.add_module('conv_{}'.format(i), conv)

            if bn == 1:
                bn = nn.BatchNorm2d(filters)
                module.add_module("bn_{}".format(i), bn)

            if active == 'leaky':
                active = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('relu_{}'.format(i), active)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear',align_corners=False)
            module.add_module('upsample_{}'.format(i), upsample)
        elif block['type'] == 'route':
            layers = block['layers']
            layers = [int(layer) for layer in layers.split(',')]

            if layers[0] > 0:
                layers[0] -= i

            if len(layers) == 1:
                filters = outputs[layers[0] + i]
            else:
                layers[1] = layers[1] - i if layers[1] > 0 else layers[1]
                filters = outputs[layers[0] + i] + outputs[layers[1] + i]

            route = EmptyLayer()
            module.add_module('route_{}'.format(i), route)
        elif block['type'] == 'shortcut':
            shorcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(i), shorcut)
        elif block['type'] == 'yolo':
            mask = block['mask']
            mask = [int(id) for id in mask.split(',')]

            anchors = block['anchors']
            anchors = [int(id) for id in anchors.split(',')]
            anchors = [(anchors[id], anchors[id + 1]) for id in range(0, len(anchors), 2)]
            anchors = [anchors[id] for id in mask]

            yolo = YoloLayer(anchors)
            module.add_module('yolo_{}'.format(i), yolo)

        prev_filters = filters
        outputs.append(filters)
        module_list.append(module)

    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = cfg_parse(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        maps = {}
        write = 0
        for i, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional' or block['type'] == 'upsample':
                x = self.module_list[i](x)
            elif block['type'] == 'route':
                layers = block['layers']
                layers = [int(layer) for layer in layers.split(',')]

                if layers[0] > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = maps[layers[0] + i]
                else:
                    layers[1] = layers[1] - i if layers[1] > 0 else layers[1]
                    x = torch.cat((maps[layers[0] + i], maps[layers[1] + i]), dim=1)
            elif block['type'] == 'shortcut':
                from_ = int(block['from'])
                x = maps[i-1] + maps[i + from_]
            elif block['type'] == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = 80
                x = prediction_transfer(x, anchors, inp_dim, num_classes)

                if not write:
                    output = x
                    write = 1
                else:
                    output = torch.cat((output, x), 1)

            maps[i] = x

        return output

    def load_weights(self, weights_file):
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)

        weights = np.fromfile(fp, dtype=np.float32)
        pr = 0

        for i, module in enumerate(self.module_list):
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional':
                conv = module[0]
                try:
                    bn = int(self.blocks[i+1]['batch_normalize'])
                except:
                    bn = 0

                if bn == 1:
                    bn = module[1]
                    bn_bias_size = bn.bias.numel()

                    bn_bias = torch.from_numpy(weights[pr : pr + bn_bias_size])
                    pr += bn_bias_size

                    bn_weights = torch.from_numpy(weights[pr : pr + bn_bias_size])
                    pr += bn_bias_size

                    bn_running_mean = torch.from_numpy(weights[pr : pr + bn_bias_size])
                    pr += bn_bias_size

                    bn_running_var = torch.from_numpy(weights[pr : pr + bn_bias_size])
                    pr += bn_bias_size

                    bn_bias = bn_bias.view_as(bn.bias)
                    bn_weights = bn_weights.view_as(bn.weight)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.detach().copy_(bn_bias)
                    bn.weight.detach().copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    conv_bias_size = conv.bias.numel()

                    conv_bias = torch.from_numpy(weights[pr : pr + conv_bias_size])
                    pr += conv_bias_size

                    conv_bias = conv_bias.view_as(conv.bias)

                    conv.bias.detach().copy_(conv_bias)

                conv_weight_size = conv.weight.numel()

                conv_weight = torch.from_numpy(weights[pr : pr + conv_weight_size])
                pr += conv_weight_size

                conv_weight = conv_weight.view_as(conv.weight)

                conv.weight.detach().copy_(conv_weight)

if __name__ == '__main__':
    blocks = cfg_parse('./cfg/yolov3.cfg')
    net_info, module_list = create_modules(blocks)
    img = read_img('./dog-cycle-car.png')
    model = Darknet('cfg/yolov3.cfg')
    model.load_weights('./yolov3.weights')
    model.eval()
    out = model(img)
    out = write_results(out, 0.5, 0.4)
    print(out)
