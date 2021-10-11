import paddlex as pdx
import time

image_name = 'dataset/JPEGImages/4e8e7a28c.jpg'
model = pdx.load_model('dataset/unet/best_model')
start_time = 0
for i in range(200):
    if i == 100:
        start_time = time.time()
    result = model.predict(image_name)
    pdx.seg.visualize(image_name, result, weight=0.4, save_dir='output/unet')

fps = 1 / (time.time() - start_time) / 100
