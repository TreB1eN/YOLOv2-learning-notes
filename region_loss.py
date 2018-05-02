import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import pdb

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    """
    pred_boxes.shape = [27040, 4] (32 * 5 * 13 * 13 = 27040)
    target.shape = [32, 250]
    anchors [十个anchor尺寸]
    num_anchors = 5
    num_classes = 20
    nH = 13
    nW = 13
    noobject_scale = 1.0
    object_scale = 5.0

    We use sum-squared error because it is easy to optimize,
     however it does not perfectly align with our goal of
    maximizing average precision. 
    It weights localization error equally with classification error which may not be ideal.
    Also, in every image many grid cells do not contain any
    object. This pushes the “confidence” scores of those cells
    towards zero, often overpowering the gradient from cells
    that do contain objects. This can lead to model instability,
    causing training to diverge early on.
    To remedy this, we increase the loss from bounding box
    coordinate predictions and decrease the loss from confi-
    dence predictions for boxes that don’t contain objects. We
    use two parameters, λcoord and λnoobj to accomplish this. We
    set λcoord = 5 and λnoobj = .5

    sil_thresh = 0.6
    seen = 32 (这个数值会变)
    """
    nB = target.size(0)
    # nB = 32
    nA = num_anchors
    # nA = 5
    nC = num_classes
    # nC = 20
    # anchor_step = len(anchors)/num_anchors
    anchor_step = len(anchors)//num_anchors
    # python3的改动，anchor_step = 2
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        # cur_pred_boxes.shape [4, 845],从pred_boxes里按顺序取每一张图片对应的845个预测的boxes
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            """
            g代表ground truth,这里的操作是跟dataset load出来的target的数据格式相关
            target是[class_id1,x1,y1,w1,h1,class_id2,x2,y2,w2,h2,,,,,]不足补零到250
            这里相当于在遍历所有ground truth的box
            gx,gy,gw,gh = (5.468634686346863, 5.45223880597015, 5.276752767527675, 10.749253731343284)
            """
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            # shape = [4, 845] repeat845遍为了下来计算IoUs
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            # 注意这个torch.max是在循环里的，所以cur_ious这个tensor是在反复地和不同的gt box作max
            # 这里没有考虑class,所以最终得到的是cur_pred_boxes里每一个box和所有gt box求IoU后所能得到的最大IoU的值
        conf_mask[b][cur_ious>sil_thresh] = 0
        """
        首先，默认的conf_mask是1,代表是背景的一个权重。
        这里是对所有最大IoU大于阈值的predicted box都先标记0,
        后面会在这些标记了的box里面找出最匹配的几个打上标签，再乘以前景的权重。
        那些这里标记为0，后面又没被选上的地方可以理解是一些次优解。
        标记为0就表示放弃对他们的计算了
        想一想如果不这么做会怎么样 ？
        有一些明明对上了一部分的点，也被当做没对上参与回归了
        或者只对上了一部分的点，也被当做完全对上了参与计算
        YOLOv3 predicts an objectness score for each bounding box 
        using logistic regression. This should be 1 if the bounding box prior 
        overlaps a ground truth object by more than any other bounding box prior. 
        If the bounding box prior is not the best but does overlap a ground truth object 
        by more than some threshold we ignore the prediction, following [15].
        We use the threshold of .5. 
        Unlike [15] our system only assigns one bounding box prior for each ground truth object. 
        If a bounding box prior is not assigned to a ground truth object
        it incurs no loss for coordinate or class predictions, only objectness.
        """
    if seen < 12800:
        #12800是一个epoch中的图片总数吧 ？
       if int(anchor_step) == 4:
            # 一路下来用的anchor_step都是2
           tx = torch.FloatTensor(anchors).view(nA, int(anchor_step)).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, int(anchor_step)).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5) # [32, 5, 13, 13]
           ty.fill_(0.5) # [32, 5, 13, 13]
       tw.zero_() # [32, 5, 13, 13]
       th.zero_() # [32, 5, 13, 13]
       coord_mask.fill_(1) # [32, 5, 13, 13]
       """
       这一段也值得好好理解
       seen代表总共见过多少图片
       seen < 某个值表示训练的初始阶段
       可以看看在初始阶段改了哪些东西
       首先，coord_mask被全部置1（本来全0）
       coord_mask在后面会被用来计算x,y,w,h的loss,这也是为什么取名叫坐标掩码
       tx,ty都用0.5填充（本来是0）
       th,tw都用0填充(本来也是0)
       在下面一大段循环中，会根据ground truth给tx,ty,th,tw中匹配上的地方改成正确的值
       但是每一张图片，生成13*13*5个anchor只有少数几个能匹配上
       剩下的可以都用掩码0给取消掉，不参与计算
       训练后期的确是这样的
       但是训练初期，网络很大程度上是在随机的输出坐标，
       与其让那些没匹配上的地方置0，
       不如告诉machine，所有瞎猜的地方，你就猜x=y=0.5
       这意味着你反正先尽量的往网格的中间猜，这样猜中的几率才大
       同时猜th=tw=0,
       因为如果匹配到了gt box，th和tw在后面会被改成实际匹配的大小的
       剩下的都是没匹配到的网格，也就是这个网格里只有背景
       相当于告诉machine,那些没有object里，你也不要找了，
       预测th=tw=0意味着预测一个无穷小的点，实际上就是虚无了
       主要是可以和那些前景th,tw有实际值的case形成强烈对比
       这样可以在初期让网络加速学到东西
       当网络看过这么多图片之后，这段代码就不起作用了
       """

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            # 这两层循环遍历的是每一张batch里面的图片以及该张图片对应的每一个ground truth box
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            # 注意这里乘以nW和nH,是为了把[0,1]之间的坐标和长宽放大到13:13的比例上
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            """
            这里取整相当于把ground truth box的中心归类到13*13个方框中的一个里面
            选中了这个位移对应的方框再去找最匹配的anchor尺寸,gi,gj后面会被反复取用
            """
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                # 然后遍历5个anchor中的每一对的w和w
                aw = anchors[int(anchor_step)*n]
                ah = anchors[int(anchor_step)*n+1]
                # anchors : [十个anchor尺寸],这里是把anchors的5对width和height逐一取出
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                # 不考虑中心点，仅比较h和w,是为了从5种选择中得到最佳的尺寸？
                if int(anchor_step) == 4:
                    ax = anchors[int(anchor_step)*n+2]
                    ay = anchors[int(anchor_step)*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                    # 这个for循环的作用就是记录下了最大的IoU和最大IoU对应的anchor在预设anchor列表中的序号
                    # 这个小for循环寻找的是，基于一张图片的一个ground truth box，找到尺寸上最匹配的预设anchor(仅仅是尺寸)
                elif int(anchor_step)==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            """
            注意这里indent的位置，是在遍历每一张图片的每一个ground truth
            还有best_n是从前面一个for循环里拿出来的，是当前5个预设anchor scales中最匹配ground truth的那个
            所以下面设计num_anchor=5这个维度的都只是从中取了最优的那个值
            """
            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi] # size : [4]
            # b*nAnchors+best_n*nPixels+gj*nW+gi 合起来其实就是找最近pred_box的坐标，只不过高维的序号被拍扁到1维
            coord_mask[b][best_n][gj][gi] = 1 # 每一个gt box对应的小方框 + 最匹配gt box的预设anchor尺寸
            cls_mask[b][best_n][gj][gi] = 1 # 每一个gt box对应的小方框 + 最匹配gt box的预设anchor尺寸
            conf_mask[b][best_n][gj][gi] = object_scale # 每一个gt box对应的小方框 + 最匹配gt box的预设anchor尺寸，打底是1，object_scale是5
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi # 该gt box的左上角在这个小方框当中的偏移量(方框的左上角是(0,0)) + 最匹配gt box的预设anchor尺寸
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj # 该gt box的左上角在这个小方框当中的偏移量(方框的左上角是(0,0)) + 最匹配gt box的预设anchor尺寸
            tw[b][best_n][gj][gi] = math.log(gw/anchors[int(anchor_step)*best_n]) #套论文上的回归公式
            th[b][best_n][gj][gi] = math.log(gh/anchors[int(anchor_step)*best_n+1]) #套论文上的回归公式
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # 加上中心点后再算一次IoU
            tconf[b][best_n][gj][gi] = iou #用final IoU作为true confidence target
            tcls[b][best_n][gj][gi] = target[b][t*5] #target类别，每5个值的第一个就是
            if iou > 0.5:
                nCorrect = nCorrect + 1 
            """
            总结一下，build_targets的核心思想是把ground truth box映射到相对应的小方框(13*13)中，
            求得在这个小方框内的偏移，就是x,y坐标上要拟合的目标。
            论文里说这样可以避免在训练开始阶段想，预测的x,y坐标四处乱飞的情况
            如果小方框定了，那就可以先不管x,y坐标，假设x,y都是0，从5个预设的anchor尺寸挑一个最接近ground truth的
            这样x,y,w,h就都生成了，剩下的就是一些补充计算和记录工作了
            注意返回的这几个变量的shape和默认值:
            conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
            coord_mask = torch.zeros(nB, nA, nH, nW) # 匹配上了就置1
            cls_mask   = torch.zeros(nB, nA, nH, nW) # 匹配上了就置1
            tx         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的offset_x
            ty         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的offset_y
            tw         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的tw
            th         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的th
            tconf      = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框与真实gt box的IoU
            tcls       = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框对应的gt class
            """
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        """
            初始化后的值
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
        self.num_anchors = num_anchors
        # num_anchors = 5
        self.anchor_step = int(len(anchors)/num_anchors)
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        """
        nB,nA,nC,nH,nW
        (32, 5, 20, 13, 13)
        """
        # output.shape : [32, 125, 13, 13]
        output   = output.view(nB, nA, (5+nC), nH, nW) # [32, 5, 25, 13, 13]
        x = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([0]))).view(nB, nA, nH, nW))
        """
        index_select : http://pytorch.org/docs/master/torch.html?highlight=index_select#torch.index_select
        相当于tf.gather
        x.shape = [32, 5, 13, 13]
        """
        y = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([1]))).view(nB, nA, nH, nW))
        # y.shape = [32, 5, 13, 13]
        w = output.index_select(2, Variable(torch.LongTensor([2]))).view(nB, nA, nH, nW)
        # w.shape = [32, 5, 13, 13]
        h = output.index_select(2, Variable(torch.LongTensor([3]))).view(nB, nA, nH, nW)
        # h.shape = [32, 5, 13, 13]
        conf = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([4]))).view(nB, nA, nH, nW))
        # conf.shape = [32, 5, 13, 13]
        cls = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long()))
        # torch.linspace(5,5+nC-1,nC).long() = [5,6,7.....,24] 取剩下20个输出 cls.shape = [32, 5, 20, 13, 13]
        cls = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        """
        注意这样和直接.view(nB*nA*nH*nW, nC)得到的结果是不同的
        这样操作的结果是以nC维度为轴，做的转换
        cls.shape = [27040, 20]
        """
        t1 = time.time()

        pred_boxes = torch.FloatTensor(4, nB*nA*nH*nW)
        # pred_boxes.shape = [4, 27040]
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW)
        """
        grid_x.shape = [27040]
        grid_y.shape = [27040]
        生成了0,1,2,3,4,5,6,7,8,9,10,11,12,0,1,2,3,,,,,,,,,(不停的重复13*batch*5次)
        """
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1]))
        """
        把anchors分成两拨，实际上是square的两条边的大小
        self.anchors:10个items的list
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
         anchor_w : shape:[5,1]
          1.3221
          3.1927
          5.0559
          9.4711
         11.2364
         anchor_h : shape:[5,1]
          1.7314
          4.0094
          8.0989
          4.8405
         10.0071
        """
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        # 把[5,1] repeat很多遍，尺寸[27040]
        # pred_boxes[0] = x.data + grid_x
        # pred_boxes[1] = y.data + grid_y
        # pred_boxes[2] = torch.exp(w.data) * anchor_w
        # pred_boxes[3] = torch.exp(h.data) * anchor_h

        pred_boxes[0] = x.data.view([-1]) + grid_x
        pred_boxes[1] = y.data.view([-1]) + grid_y
        pred_boxes[2] = torch.exp(w.data.view([-1])) * anchor_w
        pred_boxes[3] = torch.exp(h.data.view([-1])) * anchor_h
        """
        上面这个写法太不规范，改了一下，最终输出结果是一样的
        可以这样理解这个操作,先看一下
        grid_y[13*13:13*13*2] 0,1,2,3,4,5,6,7,8,9,10,11,12,0,1,2,3,4,,,,,
        grid_x[13*13:13*13*2] 0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,,,,
        grid_x和grid_y拼起来就是每13*13一段，拼起来正好是遍历了13*13棋盘中的每个点
        x和y是经过sigmoid之后的输出，值在[0,1]之间
        意味着x+grid_x,y+grid_y绝对不会超出这个方框的范围
        x,y指的是预测出来的中心点在方框内的offset
        论文里说与其拟合绝对的位置，不如只拟合方框中的位移，
        这样可以避免训练初期x,y的坐标四处乱飞的情况
        anchor_h[13*13:13*13*2]和anchor_h[13*13:13*13*2]里面的值都只有一个(h和w还是不同的)
        一个h和一个w合起来就是一个anchor的尺寸
        每13*13个中心点共一个anchor尺寸
        总共有batch * num_anchors = 32 * 5 = 160个尺寸
        所以和起来是遍历了每一张图片(batch=32)的13*13个中心点中的每一个点上面的5种anchor尺寸，
        x和y是输出的中点，区间[0,1](经过sigmoid)
        可以看做是把每一张图划分成13*13个小格，x和y就是该小格预测出来的目标的中心点相对于小格内的坐标
        但是这个奇怪的anchor尺寸是怎么来的 ？
        难道是论文里Dimension Clusters那一段说的通过k-means从dataset里面算出来的 ？看起来是的
        """

        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        # 转到cpu也是为了截断梯度 ？
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        """
        nGT, nCorrect, coord_mask.shape, conf_mask.shape, cls_mask.shape, tx.shape, ty.shape, tw.shape, th.shape, tconf.shape,tcls.shape
        上一层输出：
        (83, # nGT : 这一个batch里总共经过了多少个ground truth object
         41, # nCorrect : predicted的anchors里，有多少是和ground truth IoU > 0.5的
         conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
         coord_mask = torch.zeros(nB, nA, nH, nW) # 匹配上了就置1，训练初期默认全都是1，中后期默认是0
         cls_mask   = torch.zeros(nB, nA, nH, nW) # 匹配上了就置1
         tx         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的offset_x，训练初期默认0.5，中后期默认被coord_mask屏蔽
         ty         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的offset_y，训练初期默认0.5，中后期默认被coord_mask屏蔽
         tw         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的tw，训练初期默认0，中后期默认被coord_mask屏蔽
         th         = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框里的th，训练初期默认0，中后期默认被coord_mask屏蔽
         tconf      = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框与真实gt box的IoU
         tcls       = torch.zeros(nB, nA, nH, nW) # 匹配上的小方框对应的gt class
        """

        cls_mask = (cls_mask == 1) # 掩码来了
        nProposals = int((conf > 0.25).sum().data[0]) # conf是网络输出的对IoU的预测，取大于0.25的所有个数，比如148

        tx    = Variable(tx)
        ty    = Variable(ty)
        tw    = Variable(tw)
        th    = Variable(th)
        tconf = Variable(tconf)
        # tcls  = Variable(tcls.view(-1)[cls_mask].long()) 
        tcls  = Variable(tcls[cls_mask].long()) # 改了一下，便于理解，输出是一致的

        coord_mask = Variable(coord_mask)
        conf_mask  = Variable(conf_mask.sqrt())
        # 0和1开根号数值都不变，实际上就是把前景的权重（5）给开根号变小了呗，为什么前面赋值的时候不直接改 ？
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC))
        # 把shape[32,5,13,13]拉成[27040]然后repeat nC=20 份，最终shape = [27040, 20]
        cls        = cls[cls_mask].view(-1, nC)  #cls.shape = [87, 20] (87个是这次match了87个)

        t3 = time.time()

        """
        coord_scale = 1.0
        class_scale = 1.0
        size_average=False 表示求MSELoss的和之后并没有除以bacth_size
        可以理解，这里每一个(x,y,w,h)都是一个单独的预测框，都是重要的
        """
        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        # 不是说要用focal loss的嘛 ？这里还是没用哦
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss
