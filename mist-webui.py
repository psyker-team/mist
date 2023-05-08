import numpy as np
import gradio as gr
from mist_v3 import init, infer
from mist_utils import load_image_from_path, closing_resize
import os
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps


def reverse_mask(mask):
    r, g, b, a = mask.split()
    mask = PIL.Image.merge('RGB', (r, g, b))
    return ImageOps.invert(mask)


config = init()
target_image_path = os.path.join(os.getcwd(), 'MIST.png')


def process_image(image, eps, steps, input_size, rate, mode, block_mode, no_resize):
    print('Processing....')
    if mode == 'Textural':
        mode_value = 1
    elif mode == 'Semantic':
        mode_value = 0
    elif mode == 'Fused':
        mode_value = 2
    if image is None:
        raise ValueError

    processed_mask = reverse_mask(image['mask'])

    image = image['image']

    print('tar_img loading fin')
    config['parameters']['epsilon'] = eps / 255.0 * (1 - (-1))
    config['parameters']['steps'] = steps

    config['parameters']["rate"] = 10 ** (rate + 3)

    config['parameters']['mode'] = mode_value
    block_num = len(block_mode) + 1
    resize = len(no_resize)
    bls = input_size // block_num
    if resize:
        img, target_size = closing_resize(image, input_size, block_num, True)
        bls_h = target_size[0]//block_num
        bls_w = target_size[1]//block_num
        tar_img = load_image_from_path(target_image_path, target_size[0],
                                       target_size[1])
    else:
        img = load_image_from_path(image, input_size, input_size, True)
        tar_img = load_image_from_path(target_image_path, input_size)
        bls_h = bls_w = bls
        target_size = [input_size, input_size]
    processed_mask = load_image_from_path(processed_mask, target_size[0], target_size[1], True)
    config['parameters']['input_size'] = bls
    print(config['parameters'])
    output_image = np.zeros([input_size, input_size, 3])
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
    return output


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Image("MIST_logo.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type='pil', tool='sketch')
                    eps = gr.Slider(0, 32, step=4, value=16, label='Strength',
                                    info="Larger strength results in stronger defense at the cost of more visible noise.")
                    steps = gr.Slider(0, 1000, step=1, value=100, label='Steps',
                                      info="Larger steps results in stronger defense at the cost of more running time.")
                    input_size = gr.Slider(256, 768, step=256, value=512, label='Output size',
                                           info="Size of the output images.")

                    mode = gr.Radio(["Textural", "Semantic", "Fused"], value="Fused", label="Mode",
                                    info="See documentation for more information about the mode")

                    with gr.Accordion("Parameters of fused mode", open=False):
                        rate = gr.Slider(0, 5, step=1, value=1, label='Fusion weight',
                                         info="Higher fusion weight leads to more emphasis on \"Semantic\" ")

                    block_mode = gr.CheckboxGroup(["Low VRAM usage mode"],
                                                  info="Use this mode if the VRAM of your device is not enough. Check the documentation for more information.",
                                                  label='VRAM mode')
                    with gr.Accordion("Experimental option for non-square input", open=False):
                        no_resize = gr.CheckboxGroup(["No-resize mode"],
                                                  info="Use this mode if you donot want your image to be resized in square shape. This option is still experimental and may reduce the strength of MIST.",
                                                  label='No-resize mode')
                    inputs = [image, eps, steps, input_size, rate, mode, block_mode, no_resize]
                    image_button = gr.Button("Mist")
                outputs = gr.Image(label='Misted image')
            image_button.click(process_image, inputs=inputs, outputs=outputs)

    demo.queue().launch(share=True)
