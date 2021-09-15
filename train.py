
import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
    T.Resize(target_size=512),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = pdx.datasets.SegDataset(
                        data_dir='./dataset',
                        file_list='./dataset/train_list.txt',
                        label_list='./dataset/labels.txt',
                        transforms=train_transforms,
                        shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
                        data_dir='./dataset',
                        file_list='./dataset/val_list.txt',
                        label_list='dataset/labels.txt',
                        transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p
model = pdx.seg.DeepLabV3P(
    num_classes=num_classes, backbone='ResNet50_vd')

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=40,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_dir='output/deeplabv3p_mobilenetv2',
    use_vdl=True)