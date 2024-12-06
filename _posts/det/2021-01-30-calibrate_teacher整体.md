---
title: calibrate teacher代码解析
subtitle: 基于mmdet的sparse annotation监督代码
date: 2024-01-30 20:48:08 +0800
categories:
  - det
tags:
  - detection
  - 代码解析
published: true
image:
---
* content
{:toc}


代码结构：

```

src
├── apis
│   ├── inference.py
│   ├── __init__.py
│   └── train.py
├── core
│   ├── __init__.py
│   └── masks
│       ├── __init__.py
│       └── structures.py
├── datasets
│   ├── builder.py
│   ├── dataset_wrappers.py
│   ├── __init__.py
│   ├── pipelines
│   │   ├── formating.py
│   │   ├── geo_utils.py
│   │   ├── __init__.py
│   │   └── rand_aug.py
│   └── samplers
│       ├── __init__.py
│       └── semi_sampler.py
├── __init__.py
├── models
│   ├── cali_read_and_cali_full_100.py
│   ├── cali_read_and_cali_full.py
│   ├── cali_read_and_cali.py
│   ├── __init__.py
│   ├── multi_stream_detector.py
│   ├── retinahead_adaptnegweiht2_focaliou.py
│   ├── utils
│   │   ├── bbox_utils.py
│   │   └── __init__.py
│   └── whh_utils.py
└── utils
    ├── hooks
    │   ├── evaluation.py
    │   ├── __init__.py
    │   ├── mean_teacher.py
    │   ├── submodules_evaluation.py
    │   ├── weight_adjust.py
    │   └── weights_summary.py
    ├── __init__.py
    ├── logger.py
    ├── patch.py
    ├── signature.py
    ├── structure_utils.py
    └── vars.py

```


比较重要的是：



