import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLoss
from cfg import *
#from layers.batchnorm.bn import BN2d

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        # x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        # x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        # x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        # x = x.view(B, hs*ws*C, H/hs, W/ws)
        # python3 中会报错，int/int 是个float,改一下

        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)

        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]
        # loss取models的最后一层，是RegionLoss(）是一个很大的函数
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        # width,height = (416, 416)
        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':#yes
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            # num_anchors = 5
            self.anchor_step = self.loss.anchor_step
            # anchor_step = 2.0
            self.num_classes = self.loss.num_classes
            # num_classes = 20

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x):
        """
        这个forward其实就是死读blocks里面预先设置好的参数，然后按照参数，一层一层的走。
        在每一层forward的同时还将该层的输出保存到outputs这个字典里
        在后面route合并的过程中会从这个字典里取之前的计算结果与后面的层的输出
        看得太变扭了，肯定是因为从原版里面导出的缘故吧，pytorch正规写法不该这么写吧
        """
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        """
        输入x.shape = [8, 3, 416, 416]
        (0): Sequential(
            (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
            (leaky1): LeakyReLU(0.1, inplace)

            # 标准block，channel扩到32维,x.shape = [8, 32, 416, 416]

          )
          (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            # pooling : x.shape = [8, 32, 208, 208]

          (2): Sequential(
            (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
            (leaky2): LeakyReLU(0.1, inplace)
          )
          # 标准block，channel扩到64维,x.shape = [8, 64, 208, 208]

          (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          # pooling : x.shape = [8, 64, 104, 104]

          (4): Sequential(
            (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
            (leaky3): LeakyReLU(0.1, inplace)
          )
          # 标准block，channel扩到128维,x.shape = [8, 128, 104, 104]

          (5): Sequential(
            (conv4): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
            (leaky4): LeakyReLU(0.1, inplace)
          )
            # (1,1) conv 压缩, x.shape = [8, 64, 104, 104]

          (6): Sequential(
            (conv5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
            (leaky5): LeakyReLU(0.1, inplace)
          )
          # (3,3)又扩大回128channel x.shape = [8, 128, 104, 104]

          (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          # pooling, x.shape = [8, 128, 52, 52]

          (8): Sequential(
            (conv6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
            (leaky6): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 256, 52, 52]

          (9): Sequential(
            (conv7): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
            (leaky7): LeakyReLU(0.1, inplace)
          )
          # (1,1) conv 压缩, x.shape = [8, 128, 52, 52]

          (10): Sequential(
            (conv8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
            (leaky8): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 256, 52, 52]

          (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          # pooling, x.shape = [8, 256, 26, 26]

          (12): Sequential(
            (conv9): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
            (leaky9): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 512, 26, 26]

          (13): Sequential(
            (conv10): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn10): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
            (leaky10): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 256, 26, 26]

          (14): Sequential(
            (conv11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn11): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
            (leaky11): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 512, 26, 26]

          (15): Sequential(
            (conv12): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
            (leaky12): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 256, 26, 26]

          (16): Sequential(
            (conv13): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn13): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
            (leaky13): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 512, 26, 26]

          (17): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          # pooling, x.shape = [8, 512, 13, 13]

          (18): Sequential(
            (conv14): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn14): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
            (leaky14): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 1024, 13, 13]


          (19): Sequential(
            (conv15): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn15): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
            (leaky15): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 512, 13, 13]

          (20): Sequential(
            (conv16): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn16): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
            (leaky16): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 1024, 13, 13] 

          (21): Sequential(
            (conv17): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn17): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
            (leaky17): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 512, 13, 13]

          (22): Sequential(
            (conv18): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn18): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
            (leaky18): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 1024, 13, 13] 

          (23): Sequential(
            (conv19): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn19): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
            (leaky19): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 1024, 13, 13] 

          (24): Sequential(
            (conv20): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1.0, 1.0), bias=False)
            (bn20): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
            (leaky20): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 1024, 13, 13] 

          (25): EmptyModule(
          )
          # 执行block['type'] == 'route'分支中的代码
          # ['-9']
          # [16]
          # 结果 ： x = outputs[16]
          # x.shape = [8, 512, 26, 26]

          (26): Sequential(
            (conv21): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn21): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
            (leaky21): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 64, 26, 26] 


          (27): Reorg(
          )
          # 这一步就是调整一下tensor的shape
          # x.shape = [8, 256, 13, 13]

            Fine-Grained Features.This modified YOLO predicts
            detections on a 13 × 13 feature map. While this is sufficient for large objects, 
            it may benefit from finer grained features for localizing smaller objects. 
            Faster R-CNN and SSD both run their proposal networks at various feature maps in
            the network to get a range of resolutions. 
            We take a different approach, simply adding a passthrough layer that brings
            features from an earlier layer at 26 × 26 resolution.
            The passthrough layer concatenates the higher resolution
            features with the low resolution features by stacking adjacent features 
            into different channels instead of spatial locations, 
            similar to the identity mappings in ResNet. This turns 
            the 26 × 26 × 512 feature map into a 13 × 13 × 2048
            feature map, which can be concatenated with the original
            features. Our detector runs on top of this expanded feature
            map so that it has access to fine grained features. This gives
            a modest 1% performance increase

            这里是不是特意改的 ？ 论文里可是说512*4 = 2048啊
            这里比论文里写的多了一个降维的步骤

          (28): EmptyModule(
          )
          # 执行block['type'] == 'route'分支中的代码
          # ['-1', '-4']
          # [27, 24]
          # 结果 ： x = torch.concat(outputs[27],outputs[24])
          # x1.shape = [8, 256, 13, 13]
          # x2.shape = [8, 1024, 13, 13]
          # x.shape = [8, 1280, 13, 13]

          (29): Sequential(
            (conv22): Conv2d(1280, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn22): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
            (leaky22): LeakyReLU(0.1, inplace)
          )
          # 标准block, x.shape = [8, 1024, 13, 13]

          (30): Sequential(
            (conv23): Conv2d(1024, 125, kernel_size=(1, 1), stride=(1, 1))
          )
          # 1*1 conv,收尾了, x.shape = [8, 125, 13, 13]

          (31): RegionLoss(
          )
          # 在forward函数里好像是直接continue了，这个函数应该会在后面再调用
          # 话说好像这个网络没有用shortcut啊
          # 但是在create_network里针对这一层，会执行elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                
                [1.3221,
                 1.73145,
                 3.19275,
                 4.00944,
                 5.05587,
                 8.09892,
                 9.47112,
                 4.84053,
                 11.2364,
                 10.0071]
                 
                loss.num_classes = int(block['classes']) # 20
                loss.num_anchors = int(block['num']) # 5
                loss.anchor_step = len(loss.anchors)/loss.num_anchors # 2
                loss.object_scale = float(block['object_scale']) # 5.0
                loss.noobject_scale = float(block['noobject_scale']) # 1.0
                loss.class_scale = float(block['class_scale']) # 1.0
                loss.coord_scale = float(block['coord_scale']) # 1.0
                out_filters.append(prev_filters)
                models.append(loss)
          """
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad']) 
                
                # 下面这一行输出的是float型，torch.nn.conv2d是不接受的，改成int
                # pad = (kernel_size-1)/2 if is_pad else 0
                pad = int((kernel_size-1)/2) if is_pad else 0

                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                """
                [1.3221,
                 1.73145,
                 3.19275,
                 4.00944,
                 5.05587,
                 8.09892,
                 9.47112,
                 4.84053,
                 11.2364,
                 10.0071]
                 """
                loss.num_classes = int(block['classes'])
                # loss.num_classes = 20
                loss.num_anchors = int(block['num'])
                # loss.num_anchors = 5
                # loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.anchor_step = len(loss.anchors)//loss.num_anchors
                # 原来会产生float型，改了
                # loss.anchor_step = 2
                loss.object_scale = float(block['object_scale'])
                # 5.0
                loss.noobject_scale = float(block['noobject_scale'])
                # 1.0
                loss.class_scale = float(block['class_scale'])
                # 1.0
                loss.coord_scale = float(block['coord_scale'])
                # 1.0
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
