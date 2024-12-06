---
title: 轻松掌握 MMDetection 整体构建流程(一)
date: 2024-01-05 23:59:00 +8000
categories:
  - good_artical
tags:
  - mmlab
  - mmdetection
---

> 来源：[https://zhuanlan.zhihu.com/p/337375549](https://zhuanlan.zhihu.com/p/337375549)
## 0 摘要

大家好，今天我们将开启新的解读篇章，本系列解读主要分享 **MMDetection** 中已经复现的主流目标检测模型。

众所周知，目标检测算法比较复杂，细节比较多，难以复现，而我们推出的 MMDetection 开源框架则希望解决上述问题。目前 MMdetection 已经复现了大部分主流和前沿模型，例如 Faster R-CNN 系列、Mask R-CNN 系列、YOLO 系列和比较新的 DETR 等等，模型库非常丰富，star 接近 13k，在学术研究和工业落地中应用非常广泛。

在上述丰富模型的基础上，我们还支持非常灵活简便的扩展模式，在熟悉本框架和阅读相关说明文档后可以轻松构建不同模型，而本系列教程的目的是进一步降低大家使用和扩展框架难度，力争将 MMDetection 打造为易用易理解的主流目标检测框架。

作为系列文章的第一篇解读，本文主要是从整体框架构建角度来解析，不会涉及到具体算法和代码，希望通过本文讲解：

- MMDetection 整体构建流程和思想
- 目标检测算法核心组件划分
- 目标检测核心组件功能

由于后续 MMDetection 结构可能会有改变，本文分析的版本是 V2.7，如果后续版本有比较大的改动，会进行同步更新。

GitHub 链接：[https://github.com/open-mmlab/mmdetection](https://link.zhihu.com/?target=https%3A//github.com/open-mmlab/mmdetection) 欢迎 star ～

## 1 目标检测算法抽象流程

按照目前目标检测的发展，可以大概归纳为如下所示：

![](https://pic1.zhimg.com/80/v2-23f3f33d5ed5792e7ad55e559a6798fc_720w.webp)

注意上面仅仅写了几个典型算法而已，简单来说目标检测算法可以按照 3 个维度划分：

- **按照 stage 个数划分**，常规是 one-stage 和 two-stage，但是实际上界限不是特别清晰，例如带 refine 阶段的算法 RepPoints，实际上可以认为是1.5 stage 算法，而 Cascade R-CNN 可以认为是多阶段算法，为了简单，上面图示没有划分如此细致
- **按照是否需要预定义 anchor 划分**，常规是 anchor-based 和 anchor-free，当然也有些算法是两者混合的
- **按照是否采用了 transformer 结构划分**，目前基于 transformer 结构的目标检测算法发展迅速，也引起了极大的关注，所以这里特意增加了这个类别的划分

不管哪种划分方式，其实都可以分成若干固定模块，然后通过模块堆叠来构建整个检测算法体系。
## 2 MMDetection 整体构建流程和思想

基于目前代码实现，所有目标检测算法都按照以下流程进行划分：

![](https://pic1.zhimg.com/80/v2-7ecc8e5e19c59a3e6682c5e3cdc34918_720w.webp)

上述流程对应 MMDetection 代码构建流程，理解每个组件的作用不仅仅对阅读算法源码有帮助，而且还能够快速理解新提出算法对应的改进部分。下面对每个模块进行详细解读。

### 2.1 训练核心组件

训练部分一般包括 9 个核心组件，总体流程是：

1. 任何一个 batch 的图片先输入到 backbone 中进行特征提取，典型的骨干网络是 ResNet
2. 输出的单尺度或者多尺度特征图输入到 neck 模块中进行特征融合或者增强，典型的 neck 是 FPN
3. 上述多尺度特征最终输入到 head 部分，一般都会包括分类和回归分支输出
4. 在整个网络构建阶段都可以引入一些即插即用增强算子来增加提取提取能力，典型的例如 SPP、DCN 等等
5. 目标检测 head 输出一般是特征图，对于分类任务存在严重的正负样本不平衡，可以通过正负样本属性分配和采样控制
6. 为了方便收敛和平衡多分支，一般都会对 gt bbox 进行编码
7. 最后一步是计算分类和回归 loss，进行训练
8. 在训练过程中也包括非常多的 trick，例如优化器选择等，参数调节也非常关键

注意上述 9 个组件不是每个算法都需要的，下面详细分析。

### 2.1.1 Backbone

![](https://pic2.zhimg.com/80/v2-cdee2bd9f289d650ddbcbd748c4be0f9_720w.webp)

backbone 作用主要是特征提取。目前 MMDetection 中已经集成了大部分骨架网络，具体见文件：`mmdet/models/backbones`，V2.7 已经实现的骨架如下：


```python

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
    'ResNeSt', 'TridentResNet'
]

```


最常用的是 ResNet 系列、ResNetV1d 系列和 Res2Net 系列。如果你需要对骨架进行扩展，可以继承上述网络，然后通过注册器机制注册使用。一个典型用法为：


```python

# 骨架的预训练权重路径
pretrained='torchvision://resnet50',
backbone=dict(
    type='ResNet', # 骨架类名，后面的参数都是该类的初始化参数
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True), 
    norm_eval=True,
    style='pytorch'),

```


通过 MMCV 中的注册器机制，你可以通过 dict 形式的配置来实例化任何已经注册的类，非常方便和灵活。

### 2.1.2 Neck

![](https://pic1.zhimg.com/80/v2-f0975c00a32fa03a80860f9c09234bbc_720w.webp)

neck 可以认为是 backbone 和 head 的连接层，主要负责对 backbone 的特征进行高效融合和增强，能够对输入的单尺度或者多尺度特征进行融合、增强输出等。具体见文件：`mmdet/models/necks`，V2.7 已经实现的 neck 如下：


```python

__all__ = [
    'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
    'NASFCOS_FPN', 'RFP', 'YOLOV3Neck'
]

```


最常用的应该是 FPN，一个典型用法是：


```python

neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048], # 骨架多尺度特征图输出通道
    out_channels=256, # 增强后通道输出
    num_outs=5), # 输出num_outs个多尺度特征图

```


### 2.1.3 Head

![](https://pic2.zhimg.com/80/v2-fdd9a6232e62c75b143153dab8ba9bc1_720w.webp)

目标检测算法输出一般包括分类和框坐标回归两个分支，不同算法 head 模块复杂程度不一样，灵活度比较高。在网络构建方面，理解目标检测算法主要是要理解 head 模块。

MMDetection 中 head 模块又划分为 two-stage 所需的 RoIHead 和 one-stage 所需的 DenseHead，也就是说所有的 one-stage 算法的 head 模块都在`mmdet/models/dense_heads`中，而 two-stage 算法还包括额外的`mmdet/models/roi_heads`。

目前 V2.7 中已经实现的 dense_heads 包括：


```python

__all__ = [
    'AnchorFreeHead', 'AnchorHead', 'GuidedAnchorHead', 'FeatureAdaption',
    'RPNHead', 'GARPNHead', 'RetinaHead', 'RetinaSepBNHead', 'GARetinaHead',
    'SSDHead', 'FCOSHead', 'RepPointsHead', 'FoveaHead',
    'FreeAnchorRetinaHead', 'ATSSHead', 'FSAFHead', 'NASFCOSHead',
    'PISARetinaHead', 'PISASSDHead', 'GFLHead', 'CornerHead', 'YOLACTHead',
    'YOLACTSegmHead', 'YOLACTProtonet', 'YOLOV3Head', 'PAAHead',
    'SABLRetinaHead', 'CentripetalHead', 'VFNetHead', 'TransformerHead'
]

```


几乎每个算法都包括一个独立的 head，而 roi_heads 比较杂，就不列出了。

需要注意的是：**two-stage 或者 mutli-stage 算法，会额外包括一个区域提取器 roi extractor，用于将不同大小的 RoI 特征图统一成相同大小**。

虽然 head 部分的网络构建比较简单，但是由于正负样本属性定义、正负样本采样和 bbox 编解码模块都在 head 模块中进行组合调用，故 MMDetection **中最复杂的模块就是 head**。在最后的整体流程部分会对该模块进行详细分析。

### 2.1.4 Enhance

![](https://pic3.zhimg.com/80/v2-65a706efe224f0b7ffc7f4fd7a65f2ca_720w.webp)

enhance 是即插即用、能够对特征进行增强的模块，其具体代码可以通过 dict 形式注册到 backbone、neck 和 head 中，非常方便(目前还不完善)。常用的 enhance 模块是 SPP、ASPP、RFB、Dropout、Dropblock、DCN 和各种注意力模块 SeNet、Non_Local、CBA 等。目前 MMDetection 中部分模块支持 enhance 的接入，例如 ResNet 骨架中的 plugins，这个部分的解读放在具体算法模块中讲解。

### 2.1.5 BBox Assigner

正负样本属性分配模块作用是进行正负样本定义或者正负样本分配（可能也包括忽略样本定义），正样本就是常说的前景样本（可以是任何类别），负样本就是背景样本。因为目标检测是一个同时进行分类和回归的问题，对于分类场景必然需要确定正负样本，否则无法训练。该模块至关重要，不同的正负样本分配策略会带来显著的性能差异，目前大部分目标检测算法都会对这个部分进行改进，至关重要。一些典型的分配策略如下：

![](https://pic3.zhimg.com/80/v2-12bae70e2ea2e4afb05d0d8d3f38ca56_720w.webp)

对应的代码在`mmdet/core/bbox/assigners`中，V2.7 主要包括：


```python

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'ApproxMaxIoUAssigner', 
    'PointAssigner', 'ATSSAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'HungarianAssigner'
]

```


### 2.1.6 BBox Sampler

在确定每个样本的正负属性后，可能还需要进行样本平衡操作。本模块作用是对前面定义的正负样本不平衡进行采样，力争克服该问题。一般在目标检测中 gt bbox 都是非常少的，所以正负样本比是远远小于 1 的。而基于机器学习观点：在数据极度不平衡情况下进行分类会出现预测倾向于样本多的类别，出现过拟合，为了克服该问题，适当的正负样本采样策略是非常必要的，一些典型采样策略如下：

![](https://pic4.zhimg.com/80/v2-91674a0710afadfd06a9ebd139f875fb_720w.webp)

对应的代码在`mmdet/core/bbox/samplers`中，V2.7 主要包括：


```python

__all__ = [
    'BaseSampler', 'PseudoSampler', 'RandomSampler',
    'InstanceBalancedPosSampler', 'IoUBalancedNegSampler', 'CombinedSampler',
    'OHEMSampler', 'SamplingResult', 'ScoreHLRSampler'
]

```


### 2.1.7 BBox Encoder

为了更好的收敛和平衡多个 loss，具体解决办法非常多，而 bbox 编解码策略也算其中一个，bbox 编码阶段对应的是对正样本的 gt bbox 采用某种编码变换（反操作就是 bbox 解码），最简单的编码是对 gt bbox 除以图片宽高进行归一化以平衡分类和回归分支，一些典型的编解码策略如下：

![](https://pic4.zhimg.com/80/v2-1f8d5e5e45886423df474d168452f50b_720w.webp)

  

对应的代码在`mmdet/core/bbox/coder`中，V2.7 主要包括：


```python

__all__ = [
    'BaseBBoxCoder', 'PseudoBBoxCoder', 'DeltaXYWHBBoxCoder',
    'LegacyDeltaXYWHBBoxCoder', 'TBLRBBoxCoder', 'YOLOBBoxCoder',
    'BucketingBBoxCoder'
]

```


### 2.1.8 Loss

Loss 通常都分为分类和回归 loss，其对网络 head 输出的预测值和 bbox encoder 得到的 targets 进行梯度下降迭代训练。

loss 的设计也是各大算法重点改进对象，常用的 loss 如下：

![](https://pic4.zhimg.com/80/v2-686b0b9ac6a82f9945ae454d18783227_720w.webp)

对应的代码在`mmdet/models/losses`中，V2.7 主要包括：


```python

__all__ = [
    'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'balanced_l1_loss',
    'BalancedL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss', 'GHMC',
    'GHMR', 'reduce_loss', 'weight_reduce_loss', 'weighted_loss', 'L1Loss',
    'l1_loss', 'isr_p', 'carl_loss', 'AssociativeEmbeddingLoss',
    'GaussianFocalLoss', 'QualityFocalLoss', 'DistributionFocalLoss',
    'VarifocalLoss'
]

```


可以看出 MMDetection 中已经实现了非常多的 loss，可以直接使用。

### 2.1.9 Training tricks

训练技巧非常多，常说的调参很大一部分工作都是在设置这部分超参。这部分内容比较杂乱，很难做到完全统一，目前主流的 tricks 如下所示:

![](https://pic3.zhimg.com/80/v2-569a12b6d4a20f8619a27b48d5b2fa42_720w.webp)

MMDetection 目前这部分还会继续完善，也欢迎大家一起贡献。

### 2.2 测试核心组件

测试核心组件和训练非常类似，但是简单很多，除了必备的网络构建部分外( backbone、neck、head 和 enhance )，不需要正负样本定义、正负样本采样和 loss 计算三个最难的部分，但是其额外需要一个 bbox 后处理模块和测试 trick。

### 2.2.1 BBox Decoder

训练时候进行了编码，那么对应的测试环节需要进行解码。根据编码的不同，解码也是不同的。举个简单例子：假设训练时候对宽高是直接除以图片宽高进行归一化的，那么解码过程也仅仅需要乘以图片宽高即可。其代码和 bbox encoder 放在一起，在`mmdet/core/bbox/coder`中。

### 2.2.2 BBox PostProcess

在得到原图尺度 bbox 后，由于可能会出现重叠 bbox 现象，故一般都需要进行后处理，最常用的后处理就是非极大值抑制以及其变种。

其对应的文件在`mmdet/core/post_processing`中，V2.7 主要包括：


```python

__all__ = [
    'multiclass_nms', 'merge_aug_proposals', 'merge_aug_bboxes',
    'merge_aug_scores', 'merge_aug_masks', 'fast_nms'
]

```


### 2.2.3 Testing tricks

为了提高检测性能，测试阶段也会采用 trick。这个阶段的 tricks 也非常多，难以完全统一，最典型的是多尺度测试以及各种模型集成手段，典型配置如下：


```python

dict(
    type='MultiScaleFlipAug',
    img_scale=(1333, 800),
    flip=True,
    transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ])

```


![](https://pic3.zhimg.com/80/v2-16e307727f0c3e941ec72c21f214b982_720w.webp)

### 2.3 训练测试算法流程

在分析完每个训练流程的各个核心组件后，为了方便大家理解整个算法构建，下面分析 MMDetection 是如何组合各个组件进行训练的，这里以 one-stage 检测器为例，two-stage 也比较类似。


```python

class SingleStageDetector(---):

   def __init__(...):
        # 构建骨架、neck和head
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)

  def forward_train(---): 
        # 先运行backbone+neck进行特征提取
        x = self.extract_feat(img)
        # 对head进行forward train，输出loss
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

  def simple_test(---):
        # 先运行backbone+neck进行特征提取
        x = self.extract_feat(img)
        # head输出预测特征图
        outs = self.bbox_head(x)
        # bbox解码和还原
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # 重组结果返回
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

```


以上就是整个检测器算法训练和测试最简逻辑，可以发现训练部分最核心的就是`bbox_head.forward_train`，测试部分最核心的是`bbox_head.get_bboxes`，下面单独简要分析。

### 2.3.1 bbox_head.forward_train

forward_train 是通用函数，如下所示：


```python

def forward_train(...):
    # 调用每个head自身的forward方法
    outs = self(x)
    if gt_labels is None:
        loss_inputs = outs + (gt_bboxes, img_metas)
    else:
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
    # 计算每个head自身的loss方法
    losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    # 返回
    return losses

```


对于不同的 head，虽然 forward 内容不一样，但是依然可以抽象为： `outs = self(x)`


```python

def forward(self, feats):
   # 多尺度特征图，一个一个迭代进行forward_single
   return multi_apply(self.forward_single, feats)

def forward_single(self, x):
   # 运行各个head独特的head forward方法，得到预测图
   ....
   return cls_score, bbox_pred...

```


而对于不同的 head，其 loss 计算部分也比较复杂，可以简单抽象为：`losses = self.loss(...)`


```python

def loss(...):
    # 1 生成anchor-base需要的anchor或者anchor-free需要的points
    # 2 利用gt bbox对特征图或者anchor计算其正负和忽略样本属性
    # 3 进行正负样本采样
    # 4 对gt bbox进行bbox编码
    # 5 loss计算，并返回
    return dict(loss_cls=losses_cls, loss_bbox=losses_bbox,...)

```


### 2.3.2 bbox_head.get_bboxes

get_bboxes函数更加简单


```python

def get_bboxes(...):
   # 1 生成anchor-base需要的anchor或者anchor-free需要的points
   # 2 遍历每个输出层，遍历batch内部的每张图片，对每张图片先提取指定个数的预测结果，缓解后面后处理压力；对保留的位置进行bbox解码和还原到原图尺度
   # 3 统一nms后处理
   return det_bboxes, det_labels...

```


## 3 总结

本文重点分析了一个目标检测器是如何通过多个核心组件堆叠而成，不涉及具体代码，大家只需要总体把握即可，其中最应该了解的是：**任何一个目标检测算法都可以分成 n 个核心组件，组件和组件之间是隔离的，方便复用和设计**。当面对一个新算法时候我们可以先分析其主要是改进了哪几个核心组件，然后就可以高效的掌握该算法。

另外还有一些重要的模块没有分析，特别是 dataset、dataloader 和分布式训练相关的检测代码，由于篇幅限制就不介绍了，如有需要欢迎在评论区留言。

再次欢迎大家使用 MMDetection，也非常欢迎社区贡献！

最后附上总图：
![??](/assets/img/Pasted_image_20240105012601.png)

  

  

快速指引：

[OpenMMLab：轻松掌握 MMDetection 整体构建流程(一)1507 赞同 · 137 评论文章](https://zhuanlan.zhihu.com/p/337375549)

[OpenMMLab：轻松掌握 MMDetection 整体构建流程(二)581 赞同 · 44 评论文章](https://zhuanlan.zhihu.com/p/341954021)

[OpenMMLab：轻松掌握 MMDetection 中 Head 流程253 赞同 · 28 评论文章](https://zhuanlan.zhihu.com/p/343433169)

[OpenMMLab：轻松掌握 MMDetection 中常用算法(一)：RetinaNet 及配置详解410 赞同 · 110 评论文章](https://zhuanlan.zhihu.com/p/346198300)