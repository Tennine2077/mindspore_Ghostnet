# 目录

项目简介

依赖

文件组织

数据准备

# 项目简介

本项目对GhostNet网络进行了Mindspore版本的迁移，主要包含两个部分

1、对GhostNet的原理简介

2、Mindspore模型的搭建与训练

# 依赖

Python 3.0+

Mindspore 1.8.1+

# 文件组织

```
mindspore_Ghostnet
├── CIFAR10        //数据集
└── GhostNet.ipynb //ipynb文件
```

# 数据准备

```python
# 数据集根目录
data_dir = "./CIFAR10"

# 下载解压并加载CIFAR-10训练数据集
download_train = Cifar10(path=data_dir, split="train", batch_size=4096, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

step_size = dataset_train.get_dataset_size()#TODO

# 下载解压并加载CIFAR-10测试数据集
# dataset_val = Cifar10(path=data_dir, split='test', batch_size=6, resize=32, download=True)
download_eval = Cifar10(path=data_dir, split="test", batch_size=1024, resize=32, download=True)
dataset_eval = download_eval.run()
```