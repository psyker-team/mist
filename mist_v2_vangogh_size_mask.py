from mist_v2_size_mask import init, infer, load_image_from_path, closing_resize
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import os


# Test the script with command: python mist_v2_vangogh_size_mask.py 16 100 512 1 2 1 0 1
if __name__ == "__main__":
    epsilon = int(sys.argv[1])
    steps = int(sys.argv[2])
    input_size = int(sys.argv[3])
    block_num = int(sys.argv[4])
    mode = int(sys.argv[5])
    rate = 10 ** (int(sys.argv[6]) + 3)
    mask = int(sys.argv[7])
    resize = int(sys.argv[8])

    bls = input_size//block_num
    image_dir_path = 'test/vangogh_random_size'

    target_image_path = 'MIST.png'

    config = init(epsilon=epsilon, steps=steps, mode=mode, rate=rate)
    config['parameters']["input_size"] = bls

    for img_id in os.listdir(image_dir_path):
        image_path = os.path.join(image_dir_path, img_id)

        if resize:
            img, target_size = closing_resize(image_path, input_size, block_num)
            bls_h = target_size[0]//block_num
            bls_w = target_size[1]//block_num
            tar_img = load_image_from_path(target_image_path, target_size[0],
                                           target_size[1])
        else:
            img = load_image_from_path(image_path, input_size)
            tar_img = load_image_from_path(target_image_path, input_size)
            bls_h = bls_w = bls
            target_size = [input_size, input_size]
        output_image = np.zeros([target_size[1], target_size[0], 3])
        if mask:
            mask_path = './test/processed_mask.png'
            processed_mask = load_image_from_path(mask_path, target_size[0], target_size[1])
        else:
            processed_mask = None
        for i in tqdm(range(block_num)):
            for j in tqdm(range(block_num)):
                if processed_mask is not None:
                    input_mask = Image.fromarray(np.array(processed_mask)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                else:
                    input_mask = None
                img_block = Image.fromarray(np.array(img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                tar_block = Image.fromarray(np.array(tar_img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])

                output_image[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h] = infer(img_block, config, tar_block, input_mask)
            output = Image.fromarray(output_image.astype(np.uint8))
            class_name = ''
            for i in range(5):
                class_name += (sys.argv[i+1] + '_')
            if mode == 2:
                class_name += (sys.argv[6] + '_')
            class_name += (sys.argv[7] + '_' + sys.argv[8])
            output_path_dir = os.path.join('test', 'vangogh_' + class_name)
            if not os.path.exists(output_path_dir):
                os.mkdir(output_path_dir)
            output_path = os.path.join(output_path_dir, img_id)
            output.save(output_path)
