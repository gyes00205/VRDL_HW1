# VRDL HW1

## 1. Specification of dependencies
I train and inference my model in **colab** environment. Because my model included **Efficientnet b2~b4** and **RegNet**, they only provided by **torchvision>=0.11.0**, so we need to install new version of torch and torchvision.
```
pip install torch==1.10.0
pip install torchvision==0.11.1
```

## 2. Training code
You can download my models in this [link](https://drive.google.com/drive/folders/1PuWjKZsGxGZGvtphCiwwqMoa6b9DeZ11?usp=sharing). After downloading my models, you should put the 6 models in `model` directory.
Or you can use `train.py` to train the model that I have trained. There are 6 models included **EfficientNet b2**, **EfficientNet b3**, **EfficientNet b4**, **ResNeXt101**, **ResNet50**, **RegNet**.
* For EfficientNet b2:
`python train.py --model=efficientnet_b2`
* For EfficientNet b3:
`python train.py --model=efficientnet_b3`
* For EfficientNet b4:
`python train.py --model=efficientnet_b4`
* For ResNeXt101:
`python train.py --model=resnext101`
* For ResNet50:
`python train.py --model=resnet50`
* For RegNet:
`python train.py --model=regnet`

And model checkpoint would be saved in `model` directory.
```
VRDL_HW1
└───model # saved checkpoint
    ├───efficient_b2_batch4_epoch100.pth
    ├───efficient_b3_batch4_epoch100.pth
    ├───efficient_b4_batch4_epoch100.pth
    ├───resnet50_batch4_epoch100_best.pth
    ├───resnext101_batch4_epoch100_best.pth
    └───regnet_x_8gf_batch4_epoch100_best.pth
```

## 3. Evaluation code
I split the training data into 8 : 2 in `train.py`. 80% of data are used to train model, 20% of data are used to evaluation. Therefore, when training the model, you would see the result as below.
```
epoch: 47 / 100
Training epoch: 47 / loss: 0.527 | acc: 0.875
Validation acc: 0.690
epoch: 48 / 100
Training epoch: 48 / loss: 0.504 | acc: 0.890
Validation acc: 0.687
```

## 4. Pre-trained models
You can download my models in this [link](https://drive.google.com/drive/folders/1PuWjKZsGxGZGvtphCiwwqMoa6b9DeZ11?usp=sharing). After downloading my models, you should put the 6 models in `model` directory.
```
VRDL_HW1
└───model # saved checkpoint
    ├───efficient_b2_batch4_epoch100.pth
    ├───efficient_b3_batch4_epoch100.pth
    ├───efficient_b4_batch4_epoch100.pth
    ├───resnet50_batch4_epoch100_best.pth
    ├───resnext101_batch4_epoch100_best.pth
    └───regnet_x_8gf_batch4_epoch100_best.pth
```

## 5. Results and Inference
You can use `inference.py` to predict the answer of image, and save your answer int `answer.txt`.
* For ensemble 6 models I have trained:
`python inference.py --ensemble=1`
Submit to Codalab would get accuracy: 0.759644
* For ResNeXt101 model:
`python inference.py --model=resnext101`
Submit to Codalab would get accuracy: 0.695021
* For ResNet50 model:
`python inference.py --model=resnet50`
Submit to Codalab would get accuracy: 0.647214
* For RegNet model:
`python inference.py --model=regnet`
Submit to Codalab would get accuracy: 0.679525
* For EfficientNet b2 model:
`python inference.py --model=efficientnet_b2`
Submit to Codalab would get accuracy: 0.624794
* For EfficientNet b3 model:
`python inference.py --model=efficientnet_b3`
Submit to Codalab would get accuracy: 0.655457
* For EfficientNet b4 model:
`python inference.py --model=efficientnet_b4`
Submit to Codalab would get accuracy: 0.637652

| Model name         | CodaLab Accuracy |
| ------------------ |------------------|
| EfficientNet b2    | 0.624794         |
| EfficientNet b4    | 0.637652         |
| ResNet 50          | 0.647214         |
| EfficientNet b3    | 0.655457         |
| RegNet             | 0.679525         |
| ResNeXt 101        | 0.695021         |
| ensemble 6 models  | 0.759644         |
