---
title: mmdet based faster rcnn解析
subtitle: 
date: 2024-03-27 23:10:40 +0800
categories: 
tags: 
published: true
image:
---
* content
{:toc}


# Model 组件
## Backbone

```python

model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',  ## 使用 pytorch 提供的在 imagenet 上面训练过的权重作为预训练权重
    # 骨架网络类名
    backbone=dict(
        # 表示使用 ResNet50
        type='ResNet',
        depth=50,
        # ResNet 系列包括 stem+ 4个 stage 输出
        num_stages=4,
        # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要，
        # 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        out_indices=(0, 1, 2, 3),
        # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        frozen_stages=1,  # 具体参见文章后面备注
        # 所有的 BN 层的可学习参数都不需要梯度，也就不会进行参数更新
        norm_cfg=dict(type='BN', requires_grad=True),
        # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新，控制整个backbone归一化算子是否需要编程eval模式
        norm_eval=True,
        # 默认采用 pytorch 模式
        style='pytorch'),

```

> [!NOTE]
> 虽然resnet常规来说是有5层的，但是一般第一层，也就是第一个卷积是没人用的


### 关于style说明
`style='caffe'` 和 `style='pytorch'` 的差别就在 `Bottleneck` 模块中

`Bottleneck` 是标准的 1x1-3x3-1x1 结构，考虑 stride=2 下采样的场景，caffe 模式下，stride 参数放置在第一个 1x1 卷积上，而 Pyorch 模式下，stride 放在第二个 3x3 卷积上：

出现两种模式的原因是因为 ResNet 本身就有不同的实现，torchvision 的 resnet 和早期 release 的 resnet 版本不一样，使得目标检测框架在使用 Backbone 的时候有两种不同的配置，不过目前新网络都是采用 PyTorch 模式


## FPN

```python

neck=dict(
        type='FPN',
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        # FPN 输出特征图个数
        num_outs=5),

```

详细流程是：

- 将c2 c3 c4 c5 4 个特征图全部经过各自 1x1 卷积进行通道变换变成 m2~m5，输出通道统一为 256
- 从 m5 开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
- 将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
- 将新 m3 进行 2 倍最近邻上采样，然后和 m2 进行 add 操作，得到新的 m2
- 对 m5 和新的融合后的 m4 ~ m2，都进行各自的 3x3 卷积，得到 4 个尺度的最终输出 p5 ~ p2
- 将 c5 进行 3x3 且 stride=2 的卷积操作，得到 p6，目的是提供一个感受野非常大的特征图，有利于检测超大物体

故 FPN 模块实现了c2 ~ c5 4 个特征图输入，p2 ~ p6 5个特征图输出，其 strides = (4,8,16,32,64)。

> [!NOTE]
> 最后一个层通常没有indice，叫做pooling

## RPN

```python

rpn_head=dict(
        type='RPNHead',  # RPN网络类型
        in_channels=256,  # RPN网络的输入通道数，也是FPN 层输出特征图通道数
        feat_channels=256,  # 中间特征图通道数
        anchor_scales=[8],  # 生成的anchor的baselen，baselen = sqrt(w*h)，w和h为anchor的宽和高
        anchor_ratios=[0.5, 1.0, 2.0],  # 每个特征图有 3 个高宽比例
        anchor_strides=[4, 8, 16, 32, 64],  # 在每个特征层上的anchor的步长（对应于原图）
        target_means=[.0, .0, .0, .0],     # 往下看
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

```


模型可训练结构比较简单
一个卷积进行特征通道变换，加上两个输出分支即可。models/anchor_heads/rpn_head.py 具体代码如下：

```python

def _init_layers(self):
        # 特征通道变换
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        # 分类分支，类别固定是2，表示前后景分类
        # 并且由于 cls loss 是 bce，故实际上 self.cls_out_channels=1
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        # 回归分支，固定输出4个数值，表示基于 anchor 的变换值
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

```


### BBox Assigner
相比不包括 FPN 的 Faster R-CNN 算法，由于其 RPN Head 是多尺度特征图，为了适应这种变化，anchor 设置进行了适当修改，FPN 输出的多尺度信息可以帮助区分不同大小物体识别问题，每一层就不再需要不包括 FPN 的 Faster R-CNN 算法那么多 anchor 了。

<large>因此，在train cfg中设置了一系列的操作</large>

```python

rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner', # RPN网络的正负样本划分
                pos_iou_thr=0.7,       # 正样本的iou阈值
                neg_iou_thr=0.3,       # 负样本的iou阈值
                min_pos_iou=0.3,       
                match_low_quality=True, # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
                ignore_iof_thr=-1),
            sampler=dict(
	            # 随机采样
                type='RandomSampler',
                # 正负样本数量
                num=256,
                # 正样本比例
                pos_fraction=0.5,
                # 正负样本比例，负样本采样上限
                neg_pos_ub=-1,
                # 把gt也作为proposal
                add_gt_as_proposals=False),
            # 允许bbox向外扩一定像素
            allowed_border=-1,
            # 正样本权重，-1表示不改变
            pos_weight=-1,
            debug=False),

```

核心参数的具体含义是：

- num = 256 表示采样后每张图片的样本总数，`pos_fraction` 表示其中的正样本比例，具体是正样本采样 128 个，那么理论上负样本采样也是 128 个
- `neg_pos_ub` 表示负和正样本比例上限，用于确定负样本采样个数上界，例如打算采样 1000 个样本，正样本打算采样 500 个，但是可能正样本才 200 个，那么正样本实际上只能采样 200 个，如果设置 `neg_pos_ub=-1` 那么就会对负样本采样 800 个，用于凑足 1000 个，但是如果设置了 `neg_pos_ub` 比例，例如 1.5，那么负样本最多采样 200x1.5=300 个，最终返回的样本实际上不够 1000 个，默认情况 `neg_pos_ub=-1`
- `add_gt_as_proposals=True` 是防止高质量正样本太少而加入的，可以保证前期收敛更快、更稳定，属于训练技巧，在 RPN 模块设置为 False，主要用于 R-CNN，因为前期 RPN 提供的正样本不够，可能会导致训练不稳定或者前期收敛慢的问题。

其实现过程比较简单，如下所示：


```python

if self.add_gt_as_proposals and len(gt_bboxes) > 0:
    # 增加 gt 作为 proposals
    bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
    assign_result.add_gt_(gt_labels)

# 计算正样本个数
num_expected_pos = int(self.num * self.pos_fraction)
# 正样本随机采样
pos_inds = self.pos_sampler._sample_pos(
    assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
# 去重
pos_inds = pos_inds.unique()

# 计算负样本数
num_sampled_pos = pos_inds.numel()
num_expected_neg = self.num - num_sampled_pos
if self.neg_pos_ub >= 0:
   # 计算负样本个数上限
    _pos = max(1, num_sampled_pos)
    neg_upper_bound = int(self.neg_pos_ub * _pos)
    if num_expected_neg > neg_upper_bound:
        num_expected_neg = neg_upper_bound

# 负样本随机采样
neg_inds = self.neg_sampler._sample_neg(
    assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
# 去重  
neg_inds = neg_inds.unique()

```


而具体的随机采样函数如下所示：


```python

# 随机采样正样本
def _sample_pos(self, assign_result, num_expected, **kwargs):
    """Randomly sample some positive samples."""
    pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
    if pos_inds.numel() != 0:
        pos_inds = pos_inds.squeeze(1)
    if pos_inds.numel() <= num_expected:
        return pos_inds
    else:
        return self.random_choice(pos_inds, num_expected)

# 随机采样负样本
def _sample_neg(self, assign_result, num_expected, **kwargs):
    """Randomly sample some negative samples."""
    neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
    if neg_inds.numel() != 0:
        neg_inds = neg_inds.squeeze(1)
    if len(neg_inds) <= num_expected:
        return neg_inds
    else:
        return self.random_choice(neg_inds, num_expected)

```


经过随机采样函数后，可以有效控制 RPN 网络计算 loss 时正负样本平衡问题。

### bbox_coder
在 anchor-based 算法中，为了利用 anchor 信息进行更快更好的收敛，一般会对 head 输出的 bbox 分支 4 个值进行编解码操作，作用有两个：

1. 更好的平衡分类和回归分支 loss，以及平衡 bbox 四个预测值的 loss
2. 训练过程中引入 anchor 信息，加快收敛

RetinaNet 采用的编解码函数是主流的 DeltaXYWHBBoxCoder，其配置如下：

```python

rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

```

target_means 和 target_stds 相当于对 bbox 回归的 4 个 $t_x$$t_y$$t_w$$t_h$ 进行变换。在不考虑 target_means 和 target_stds 情况下，其编码公式如下：
$$
t_x^* = \frac{x-x_a}{w_a}, t_y^* = \frac{y-y_a}{h_a}; \\
t_w^* = \log{\left(\frac{w}{w_a} \right)}, t_h^*=\log \left(\frac{h}{ h_a}\right)
$$


编码过程

```python

dx = (gx - px) / pw
dy = (gy - py) / ph
dw = torch.log(gw / pw)
dh = torch.log(gh / ph)
deltas = torch.stack([dx, dy, dw, dh], dim=-1)
# 最后减掉均值，处于标准差
means = deltas.new_tensor(means).unsqueeze(0)
stds = deltas.new_tensor(stds).unsqueeze(0)
deltas = deltas.sub_(means).div_(stds)

```

解码过程是编码过程的反向，比较容易理解，其核心代码如下：


```python

# 先乘上 std，加上 mean
means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
denorm_deltas = deltas * stds + means
dx = denorm_deltas[:, 0::4]
dy = denorm_deltas[:, 1::4]
dw = denorm_deltas[:, 2::4]
dh = denorm_deltas[:, 3::4]
# wh 解码
gw = pw * dw.exp()
gh = ph * dh.exp()
# 中心点 xy 解码
gx = px + pw * dx
gy = py + ph * dy
# 得到 x1y1x2y2 的 gt bbox 预测坐标
x1 = gx - gw * 0.5
y1 = gy - gh * 0.5
x2 = gx + gw * 0.5
y2 = gy + gh * 0.5

```


### Loss

```python

loss_cls=dict(
    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

```

RPN 采用的 loss 是常用的 ce loss 和 l1 loss，不需要详细描述。


## ROIHead

```python

bbox_roi_extractor=dict(
        type='SingleRoIExtractor',                                   # RoIExtractor类型
        # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),  
        out_channels=256,                                            # 输出通道数
        featmap_strides=[4, 8, 16, 32]),                             # 特征图的步长
    bbox_head=dict(
        # 2 个共享 FC 模块
        type='SharedFCBBoxHead',
        num_fcs=2,
        # 输入通道数，相等于 FPN 输出通道
        in_channels=256,
       # 输出通道数
        fc_out_channels=1024,.
        # RoIAlign 或 RoIPool 输出的特征图大小
        roi_feat_size=7,
        # COCO数据集类别个数，这里是因为版本问题，加了一个背景，在2.0不在+1
        num_classes=80,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        # 是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        # 影响 bbox 分支的通道数，True 表示 4 通道输出，False 表示 4×num_classes 通道输出
        reg_class_agnostic=False,
        #  RPN 采用的 loss 是常用的分类 ce loss 和回归 l1 loss
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

```

这里多了一个ROI extractor

1. RPN 层输出每张图片最多 `nms_post` 个候选框，故 R-CNN 输入 shape 为 `(batch, nms_post, 4)`，4 表示 RoI 坐标。
2. 利用 RoI 重映射规则，将 `nms_post` 个候选框映射到 FPN 输出的不同特征图上，提取对应的特征图，然后利用插值思想将其变成指定的固定大小输出，输出 shape 为 `(batch, nms_post, 256, roi_feat_size, roi_feat_size)`，其中 256 是 FPN 层输出特征图通道大小，`roi_feat_size` 一般取 7。上述步骤即为 RoIAlign 或者 RoIPool 计算过程。
3. 将 `(batch, nms_post, 256, roi_feat_size, roi_feat_size)` 数据拉伸为 `(batch*nms_post, 256*roi_feat_size*roi_feat_size)`，转化为 FC 可以支持的格式, 然后应用两次共享卷积，输出 shape 为 `(batch*nms_post, 1024)`。
4. 将 `(batch*nms_post, 1024)` 分成分类和回归分支，分类分支输出 `(batch*nms_post, num_classes+1)`, 回归分支输出 `(batch*nms_post, 4*num_class)`。


$$k=\lfloor k_0 + log_2(\sqrt{wh}/224)\rfloor$$
上述公式中 $k_0$=4，通过公式可以算出 pk，具体是：
- wh>=448x448，则分配给 p5
- wh<448x448 并且 wh>=224x224，则分配给 p4
- wh<224x224 并且 wh>=112x112，则分配给 p3
- 其余分配给 p2


```python

def map_roi_levels(self, rois, num_levels):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
     - scale >= finest_scale * 8: level 3
    """
    scale = torch.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    return target_lvls

```

其中 `finest_scale=56，num_level=5`。

然后经过两层分类和回归共享全连接层 FC，最后是各自的输出头，其 forward 逻辑如下：

```python

if self.num_shared_fcs > 0:
    x = x.flatten(1)
    # 两层共享 FC
    for fc in self.shared_fcs:
        x = self.relu(fc(x))

x_cls = x
x_reg = x

# 不共享的分类和回归分支输出
cls_score = self.fc_cls(x_cls) if self.with_cls else None
bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
return cls_score, bbox_pred

```

最终输出分类和回归预测结果。相比于目前主流的全卷积模型，Faster R-CNN 的 R-CNN 模块依然采用的是全连接模式。

### 训练逻辑

```python

rcnn=dict(
    assigner=dict(
        # 和 RPN 一样，正负样本定义参数不同
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        match_low_quality=False,
        ignore_iof_thr=-1),
    sampler=dict(
        # 和 RPN 一样，随机采样参数不同
        type='RandomSampler',
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        # True，RPN 中为 False
        add_gt_as_proposals=True)

```


理论上，BBox Assigner 和 BBox Sampler 逻辑可以放置在 _(1) 公共部分_ 后面，因为其任务是输入每张图片的 `nms_post` 个候选框以及标注的 gt bbox 信息，然后计算每个候选框样本的正负样本属性，最后再进行随机采样尽量保证样本平衡。R-CNN的候选框对应了 RPN 阶段的 anchor，只不过 RPN 中的 anchor 是预设密集的，而 R-CNN 面对的 anchor 是动态稀疏的，RPN 阶段基于 anchor 进行分类回归对应于 R-CNN 阶段基于候选框进行分类回归，思想是完全一致的，故 Faster R-CNN 类算法叫做 two-stage，因此可以简化为 **one-stage** + RoI 区域特征提取 + **one-stage**。


配置参数和 RPN 不同：

- `match_low_quality=False`。为了避免出现低质量匹配情况(因为 two-stage 算法性能核心在于 R-CNN，RPN 主要保证高召回率，R-CNN 保证高精度)，R-CNN 阶段禁用了允许低质量匹配设置
- 3 个 `iou_thr` 设置都是 0.5，不存在忽略样本，这个参数在 Cascade R-CNN 论文中有详细说明，影响较大
- `add_gt_as_proposals=True`。主要是克服刚开始 R-CNN 训练不稳定情况


### 整体逻辑

```python

if self.with_bbox or self.with_mask:
    num_imgs = len(img_metas)
    sampling_results = []
    # 遍历每张图片，单独计算 BBox Assigner 和 BBox Sampler 
    for i in range(num_imgs):

        # proposal_list 是 RPN test 输出的候选框
        assign_result = self.bbox_assigner.assign(
            proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
            gt_labels[i])

        # 随机采样
        sampling_result = self.bbox_sampler.sample(
            assign_result,
            proposal_list[i],
            gt_bboxes[i],
            gt_labels[i],
            feats=[lvl_feat[i][None] for lvl_feat in x])
        sampling_results.append(sampling_result)

# 特征重映射+ RoI 区域特征提取+ 网络 forward + Loss 计算
losses = dict()
# bbox head forward and loss
if self.with_bbox:
    bbox_results = self._bbox_forward_train(x, sampling_results,
                                            gt_bboxes, gt_labels,
                                            img_metas)
    losses.update(bbox_results['loss_bbox'])

# mask head forward and loss
if self.with_mask:
    mask_results = self._mask_forward_train(x, sampling_results,
                                            bbox_results['bbox_feats'],
                                            gt_masks, img_metas)
    losses.update(mask_results['loss_mask'])
return losses

```


`_bbox_forward_train` 逻辑和 RPN 非常类似，只不过多了额外的 RoI 区域特征提取步骤：


```python

def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                        img_metas):
    rois = bbox2roi([res.bboxes for res in sampling_results])

    # 特征重映射+ RoI 特征提取+ 网络 forward
    bbox_results = self._bbox_forward(x, rois)

    # 计算每个样本对应的 target, bbox encoder 在内部进行
    bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                              gt_labels, self.train_cfg)

    # 计算 loss
    loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                    bbox_results['bbox_pred'], rois,
                                    *bbox_targets)
    bbox_results.update(loss_bbox=loss_bbox)
    return bbox_results

```


`_bbox_forward` 逻辑是 R-CNN 的重点：


```python

def _bbox_forward(self, x, rois):
    # 特征重映射+ RoI 区域特征提取，仅仅考虑前 num_inputs 个特征图
    bbox_feats = self.bbox_roi_extractor(
        x[:self.bbox_roi_extractor.num_inputs], rois)

    # 共享模块
    if self.with_shared_head:
        bbox_feats = self.shared_head(bbox_feats)

    # 独立分类和回归 head
    cls_score, bbox_pred = self.bbox_head(bbox_feats)

    bbox_results = dict(
        cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
    return bbox_results

```


### 测试逻辑

```python

rcnn=dict(
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)

```

测试逻辑核心逻辑如下：

- 公共逻辑部分输出 batch * nms_post 个候选框的分类和回归预测结果
- 将所有预测结果按照 batch 维度进行切分，然后依据单张图片进行后处理，后处理逻辑为：先解码并还原为原图尺度；然后利用 `score_thr` 去除低分值预测；然后进行 NMS；最后保留最多 `max_per_img` 个结果


# Dataset组件

```python

# dataset settings
dataset_type = 'CocoDataset'                # 数据集类型
data_root = 'data/coco/'                    # 数据集根目录
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)   # 输入图像初始化，减去均值mean并处以方差std，to_rgb表示将bgr转为rgb
data = dict(
    imgs_per_gpu=2,                # 每个gpu计算的图像数量
    workers_per_gpu=2,             # 每个gpu分配的线程数
    train=dict(
        type=dataset_type,                                                 # 数据集类型
        ann_file=data_root + 'annotations/instances_train2017.json',       # 数据集annotation路径
        img_prefix=data_root + 'train2017/',                               # 数据集的图片路径
        img_scale=(1333, 800),                                             # 输入图像尺寸，最大边1333，最小边800
        img_norm_cfg=img_norm_cfg,                                         # 图像初始化参数
        size_divisor=32,                                                   # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
        flip_ratio=0.5,                                                    # 图像的随机左右翻转的概率
        with_mask=False,                                                   # 训练时附带mask
        with_crowd=True,                                                   # 训练时附带difficult的样本
        with_label=True),                                                  # 训练时附带label
    val=dict(
        type=dataset_type,           
        ann_file=data_root + 'annotations/instances_val2017.json',         
        img_prefix=data_root + 'val2017/',   
        img_scale=(1333, 800),               
        img_norm_cfg=img_norm_cfg,           
        size_divisor=32,                     
        flip_ratio=0,                        
        with_mask=False,                     
        with_crowd=True,                     
        with_label=True),                    
    test=dict(
        type=dataset_type,                   
        ann_file=data_root + 'annotations/instances_val2017.json',         
        img_prefix=data_root + 'val2017/',   
        img_scale=(1333, 800),               
        img_norm_cfg=img_norm_cfg,          
        size_divisor=32,                    
        flip_ratio=0,                       
        with_mask=False,                    
        with_label=False,                   
        test_mode=True)) 

```


# Optimizer

```python

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)   # 优化参数，lr为学习率，momentum为动量因子，weight_decay为权重衰减因子
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))          # 梯度均衡参数
# learning policy
lr_config = dict(
    policy='step',                        # 优化策略
    warmup='linear',                      # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=500,                     # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3,                 # 起始的学习率
    step=[8, 11])                         # 在第8和11个epoch时降低学习率
checkpoint_config = dict(interval=1)      # 每1个epoch存储一次模型

```


# Log控制器

```python

# yapf:disable
log_config = dict(
    interval=50,                          # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),      # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

```


# 运行时环境（runtime）设置

```python

# runtime settings
total_epochs = 12                               # 最大epoch数
dist_params = dict(backend='nccl')              # 分布式参数
log_level = 'INFO'                              # 输出信息的完整度级别
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x' # log文件和模型文件存储路径
load_from = None                                # 加载模型的路径，None表示从预训练模型加载
resume_from = None                              # 恢复训练模型的路径
workflow = [('train', 1)]                       # 当前工作区名称

```

