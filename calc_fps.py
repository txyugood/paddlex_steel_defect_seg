import paddlex as pdx
import time
import glob

image_name = 'dataset/JPEGImages/4e8e7a28c.jpg'
image_names = glob.glob('dataset/JPEGImages/*.jpg')
model = pdx.load_model('output/unet/best_model')
start_time = 0
for i, image_name in enumerate(image_names):
    if i == 100:
        start_time = time.time()
    if i > 199:
        break

    result = model.predict(image_name)
    pdx.seg.visualize(image_name, result, weight=0.4, save_dir='output/unet')
fps = (time.time() - start_time) / 100
fps = 1 / fps
print(f"fps:{fps}")
