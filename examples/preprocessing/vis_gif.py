import imageio
import os

save_path = r"/home/fla/workspace/l5kit_data/rasterized"
img_save_path = os.path.join(save_path, "image/")

img_list = sorted(os.listdir(img_save_path))[400:3000]
gif_images = []
for path in img_list:
    gif_images.append(imageio.imread(os.path.join(img_save_path,path)))
imageio.mimsave("./sample.gif",gif_images,fps=24)
