import numpy as np
import gradio as gr
from mist_v2 import init, infer, load_image_from_path
import os
from tqdm import tqdm
from PIL import Image

config = init()
target_image_path = os.path.join(os.getcwd(), 'MIST.png')


def process_image(image, eps, steps, input_size, rate, mode, block_mode):
    print('Processing....')
    if mode == 'Textural':
        mode_value = 1
    elif mode == 'Semantic':
        mode_value = 0
    elif mode == 'Fused':
        mode_value = 2
    if image is None:
        raise ValueError
    tar_img = load_image_from_path(target_image_path, input_size)
    img = image.resize((input_size, input_size), resample=Image.BICUBIC)
    print('tar_img loading fin')
    config['parameters']['epsilon'] = eps / 255.0 * (1 - (-1))
    config['parameters']['steps'] = steps

    config['parameters']["rate"] = 10 ** (rate + 3)

    config['parameters']['mode'] = mode_value
    block_num = len(block_mode) + 1
    bls = input_size // block_num
    config['parameters']['input_size'] = bls
    print(config['parameters'])
    output_image = np.zeros([input_size, input_size, 3])
    for i in tqdm(range(block_num)):
        for j in tqdm(range(block_num)):
            img_block = Image.fromarray(np.array(img)[bls * i: bls * i + bls, bls * j: bls * j + bls])
            tar_block = Image.fromarray(np.array(tar_img)[bls * i: bls * i + bls, bls * j: bls * j + bls])
            output_image[bls * i: bls * i + bls, bls * j: bls * j + bls] = infer(img_block, config, tar_block)
    output = Image.fromarray(output_image.astype(np.uint8))
    return output


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Image("MIST_logo.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type='pil')
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
                    inputs = [image, eps, steps, input_size, rate, mode, block_mode]
                    image_button = gr.Button("Mist")
                outputs = gr.Image(label='Misted image')
            image_button.click(process_image, inputs=inputs, outputs=outputs)

    demo.queue().launch(share=True)
