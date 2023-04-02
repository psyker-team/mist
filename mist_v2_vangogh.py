from mist_v2 import init, infer, load_image_from_path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import os


if __name__ == "__main__":
    epsilon = int(sys.argv[1])
    steps = int(sys.argv[2])
    input_size = int(sys.argv[3])
    block_num = int(sys.argv[4])
    mode = int(sys.argv[5])
    rate = 10 ** (int(sys.argv[6]) + 3)

    bls = input_size//block_num
    image_dir_path = 'test/vangogh'

    target_image_path = 'MIST.png'
    img = load_image_from_path(image_path, input_size)
    tar_img = load_image_from_path(target_image_path, input_size)

    config = init(epsilon=epsilon, steps=steps, mode=mode, rate=rate)
    config['parameters']["input_size"] = bls

    output_image = np.zeros([input_size, input_size, 3])
    for i in tqdm(range(block_num)):
        for j in tqdm(range(block_num)):
            img_block = Image.fromarray(np.array(img)[bls*i: bls*i+bls, bls*j: bls*j + bls])
            tar_block = Image.fromarray(np.array(tar_img)[bls*i: bls*i+bls, bls*j: bls*j + bls])
            output_image[bls*i: bls*i+bls, bls*j: bls*j + bls] = infer(img_block, config, tar_block)
        output = Image.fromarray(output_image.astype(np.uint8))
        class_name = ''
        for i in range(5):
            class_name += (sys.argv[i+1] + '_')
        if mode == 2:
            class_name += (sys.argv[6] + '_')
        output_path_dir = os.path.join('test', class_name)
        if not os.path.exists(output_path_dir):
            os.mkdir(output_path_dir)
        output_path = os.path.join(output_path_dir, img_name)
        output.save(output_path)
