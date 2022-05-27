# Mitigating Memorization of Noisy Labels via Regularization between Representations
This code is a PyTorch implementation of the paper "[Mitigating Memorization of Noisy Labels via Regularization between Representations](https://arxiv.org/abs/2110.09022)".
The code is run on the Tesla V-100.
## Prerequisites
Python 3.8.5

PyTorch 1.7.1

Torchvision 0.9.0


## Guideline
### Downloading dataset: 

Download the dataset from **http://www.cs.toronto.edu/~kriz/cifar.html** Put the dataset on **data/**


### Get a SSL pre-trained model:

Run command below:

```
python main.py --batch_size 512 --epochs 1000 
```
Or download the pre-trained model from "[this url](https://drive.google.com/file/d/10IUG97crgC5S34kcbqtw7LOUuhPiol2V/view?usp=sharing)" and put the pre-trained model in **results/**

### CE with fixed encoder on symm. label noise:
Run command below:
```
python CE.py --simclr_pretrain --finetune_fc_only --noise_type symmetric --noise_rate 0.6 --epochs 150 
```
### CE with fixed encoder on instance label noise (with down-sampling):
Run command below:
```
python CE.py --simclr_pretrain --down_sample --finetune_fc_only --noise_type instance --noise_rate 0.6 --epochs 150 
```

### CE with regularizer on symm. label noise (random initialization):
Run command below:
```
python CE_Reg.py --reg rkd_dis --noise_type symmetric --noise_rate 0.6 --epochs 150 
```


## References

The code of SSL pre-training is based on **https://github.com/leftthomas/SimCLR**
