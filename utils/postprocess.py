import os 
import PIL
from PIL import Image


def crop_resize_from_path(input_path, input_size, target_size):
    crop = (input_size - target_size)//2
    box = [crop, crop, 512 - crop, 512-crop]
    target_path = input_path + '_crop_resize'
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for img_id in os.listdir(input_path):
        input_image_path = os.path.join(input_path, img_id)
        if os.path.isdir(input_image_path):
            continue
        img = Image.open(input_image_path).resize((input_size, input_size),
                                                  resample=PIL.Image.BICUBIC)
        img = img.crop(box).resize((input_size, input_size),
                                   resample=PIL.Image.BICUBIC)
        target_image_path = os.path.join(target_path, img_id)
        img.save(target_image_path)


if __name__ == "__main__":
    input_path = 'test/vangogh_16_100_512_1_2_1'
    crop_resize_from_path(input_path, 512, 384)