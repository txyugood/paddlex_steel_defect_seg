import paddlex as pdx

image_name = 'dataset/JPEGImages/4e8e7a28c.jpg'
model = pdx.load_model('dataset/unet/best_model')
result = model.predict(image_name)
pdx.seg.visualize(image_name, result, weight=0.4, save_dir='output/unet')