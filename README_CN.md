# ConvLSTM_MindSpore

参考论文：Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (https://doi.org/10.48550/arXiv.1506.04214)

参考git项目：

[1]	https://github.com/czifan/ConvLSTM.pytorch

[2]	https://github.com/AIS-Bonn/vp-suite

[3]	https://github.com/ndrplz/ConvLSTM_pytorch

## 代码目录结构

```shell
|- ConvLSTM						# 模型名
	|- dataset					# 数据集存放路径
		|- mnist_test_seq.npy	# 数据集
	|- output					# 训练与评估过程中输出的存放路径
		|- model				# 训练过程中保存的模型路径
			|- model.ckpt			# 训练过程中保存的模型
			|- ...
	|- src						# 模型定义源码目录
		|- config.py			# 模型配置参数文件
		|- ConvLSTM.py			# 模型结构定义
		|- dataset.py			# 数据集处理定义
		|- functions.py			# 单轮(epoch)训练操作定义
		|- OneStep.py			# 单步(step)训练操作定义
		|- SSIM.py				# SSIM计算定义
	|- eval.py					# 模型验证
	|- train.py					# 模型训练与验证
```

## 环境要求

```
硬件平台：CPU / GPU / Ascend

第三方库：
mindspore = 1.6.1
numpy = 1.20.1

数据集：mnist_test_seq.npy (https://paperswithcode.com/dataset/moving-mnist)
```

## 如何运行

执行训练：

```shell
python train.py							# 不使用预训练模型参数
python train.py --pretrain model.ckpt	# 使用预训练模型参数(该模型为第100轮保存的)
```

执行验证：

```
python eval.py
```

