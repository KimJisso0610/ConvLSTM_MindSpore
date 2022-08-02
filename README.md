# ConvLSTM_MindSpore

Reference: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (https://doi.org/10.48550/arXiv.1506.04214)

Reference git project:

[1]	https://github.com/czifan/ConvLSTM.pytorch

[2]	https://github.com/AIS-Bonn/vp-suite

[3]	https://github.com/ndrplz/ConvLSTM_pytorch

## Code structure

```shell
|- ConvLSTM						# Model name
	|- dataset					# Dataset storage path
		|- mnist_test_seq.npy	# Dataset
	|- output					# The storage path of the output during training and evaluation
		|- model				# Model path saved during training
			|- model.ckpt		# Model saved during training
			|- ...
	|- src						# Model definition source directory
		|- config.py			# Model configuration parameter file
		|- ConvLSTM.py			# Model Structure Definition
		|- dataset.py			# Dataset Processing Definition
		|- functions.py			# Single epoch training operation definition
		|- OneStep.py			# Single step training operation definition
		|- SSIM.py				# SSIM calculation definition
	|- eval.py					# Model evaluation
	|- train.py					# Model training and evaluation
```

## Environmental requirements

```
Hardware Platform: CPU / GPU / Ascend

Third Party Libraries:
mindspore = 1.6.1
numpy = 1.20.1

Dataset: mnist_test_seq.npy (https://paperswithcode.com/dataset/moving-mnist)
```

## How to run

Training:

```shell
python train.py							# Not using pretrained model parameters
python train.py --pretrain model.ckpt	# Use pretrained model parameters (the model was saved for epoch 100)
```

Evaluation:

```
python eval.py
```

