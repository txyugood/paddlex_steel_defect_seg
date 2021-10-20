import paddlex as pdx
from paddlex import transforms as T

eval_transforms = T.Compose([
    T.Resize(target_size=(64, 400)),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_dataset = pdx.datasets.SegDataset(
    data_dir='./dataset',
    file_list='./dataset/val_list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms)

model = pdx.load_model('output/unet/best_model')

metrics, evaluate_details = model.evaluate(eval_dataset, batch_size=1, return_details=True)

for key in metrics.keys():
    print(f"{key}:{metrics[key]} ", end="")