---
title: zsh对/etc/profile变量不生效问题
subtitle: 
date: 2024-04-08 01:01:14 +0800
categories:
  - Tools
tags:
  - basic
published: true
image:
---
* content
{:toc}

之前配置java环境的时候出现明明将`$PATH`环境变量导入到了profile里面，但是登录的时候怎么也找不到路径。后面一探缘由，竟然是zsh不解析`/etc/profile`


## linux解析配置文件顺序
### 正常bash
1. 非登录式Shell(如运行脚本\[tmux, bash\]、执行命令等)  
    解析顺序:  
    /etc/profile -> ~/.bashrc -> /etc/bashrc -> ~/.bash_profile
    
2. 登录式Shell(如打开终端、SSH登录等)  
    解析顺序:  
    /etc/profile -> ~/.bash_profile -> ~/.bashrc -> /etc/bashrc -> ~/.bash_logout

### zsh
zsh并不使用/etc/profile文件，而是使用/etc/zsh/下面的zshenv、zprofile、zshrc、zlogin文件，并以这个顺序进行加载。

## 解决方案
在/etc/zsh/profile里面添加

```bash

source /etc/profile

```






---
> 更详细请参考：[大佬解释](https://luckymrwang.github.io/2015/06/04/bash-profile-profile-bashrc%E7%9A%84%E5%8C%BA%E5%88%AB%E5%92%8C%E5%90%AF%E5%8A%A8%E9%A1%BA%E5%BA%8F/)
