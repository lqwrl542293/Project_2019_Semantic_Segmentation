# project

### 数据准备

把数据训练测试，以及标签分别放在../datasets/train test 以及vis的文件夹下面

标签转换，把颜色标签准换为数字

```go
python data_generate/labelTransform.py
```

对图片标签进行切割,生成图片(包含扩增代码)

```
python data_generate.py
```

### 训练

```
HRNet:
python train.py -cfg experiment/HRNet_test_gray.yaml
CCNet:
python train.py -cfg experiment/CCNet_test1.yaml
```

训练时需要修改batchsize cardnumber以及lr，optimizer等参数都可以在experiments文件夹下的配置文件中修改

目前只支持跑hrnet以及ccnet的网络

### 测试

```
HRNet:
python test.py -cfg experiment/HRNet_test_gray.yaml
CCNet:
python test.py -cfg experiment/CCNet_test1.yaml
```

需要把配置文件中的TRAIN参数换为FALSE

