# Mist-webui-v1.0

Mist is a powerful image preprocessing tool designed for the purpose of protecting the style and content of images from being mimicked by state-of-the-art AI-for-Art applications, including Stable Diffusion, NovelAI, and scenario.gg. By adding watermarks to the images, Mist renders them unrecognizable for the models employed by AI-for-Art applications. Attempts by AI-for-Art applications to mimic these Misted images will be ineffective, and the output image of such mimicry will be scrambled and unusable as artwork. 


Refer to the [document](https:) for more information.


## TODO

## Setup

Our code builds on, and shares most requirements with  [stable-diffusion](https://github.com/CompVis/stable-diffusion). In addition, [advertorch0.2.4](https://github.com/BorealisAI/advertorch) is required for performancing the attack. To set up the environment, please run: 

```
conda env create -f environments.yml
conda activate mist
pip install --force-reinstall pillow
```

The pip does not fix all the dependencies for the environment listed above and thus the Pillow may not work properly, resulting in a more visible noise. We highly recommand reinstall Pillow after you activate mist for the first time.


Official Stable-diffusion-model v1.4 checkpoint is also required, available in [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt).

Currently, the model can be downloaded by:
```
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p  models/ldm/stable-diffusion-v1
mv sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

## Usage

### Script

Use the following command to generate misted image for test/man.png:
```
python mist_v2.py 16 100 512 1 2 1
```
The misted image is saved in test/misted_man_16_100_512_1_2_1.png. 


Use the following command to generate misted image for test/vangogh dir:
```
python mist_v2_vangogh.py 16 100 512 1 2 1
```
corresponding output dir: test/vangogh_16_100_512_1_2_1. 


### Webui

Use the following command to boost webui. See [documents](https:) for more information.
```
python mist-webui.py
```


### Crop-Resize

We use the following scripts to crop and resize the misted images to test the effects of mist under post-processing. 
```
python utils/postprocess.py
```


## Validation

Refer to the [validation document](https:) to validate the effects of mist.
