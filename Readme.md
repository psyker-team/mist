# Mist-webui-v1.0

Mist is a powerful image preprocessing tool designed for the purpose of protecting the style and content of images from being mimicked by state-of-the-art AI-for-Art applications, including Stable Diffusion, NovelAI, and scenario.gg. By adding watermarks to the images, Mist renders them unrecognizable for the models employed by AI-for-Art applications. Attempts by AI-for-Art applications to mimic these Misted images will be ineffective, and the output image of such mimicry will be scrambled and unusable as artwork. 


Refer to the [document](https://mist-documentation.readthedocs.io/en/latest) for more information.


## Setup

Our code builds on, and shares most requirements with  [stable-diffusion](https://github.com/CompVis/stable-diffusion). In addition, [advertorch0.2.4](https://github.com/BorealisAI/advertorch) is required for performancing the attack. To set up the environment, please run: 

```
conda env create -f environments.yml
conda activate mist
pip install --force-reinstall pillow
```

The pip does not fix all the dependencies for the environment listed above and thus the Pillow may not work properly, resulting in a more visible noise. We highly recommand reinstall Pillow after you activate virtual environment mist for the first time.


Official Stable-diffusion-model v1.4 checkpoint is also required, available at [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt).

Currently, the model can be downloaded by:
```
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p  models/ldm/stable-diffusion-v1
mv sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

## Usage

### Running Mist with Python script

Use the following command to Mist image saved in path test/man.png:
```
python mist_v2.py 16 100 512 1 2 1
```
The Misted image is saved in test/misted_man_16_100_512_1_2_1.png. 


Use the following command to generate misted image for test/vangogh dir:
```
python mist_v2_vangogh.py 16 100 512 1 2 1
```
corresponding output dir is test/vangogh_16_100_512_1_2_1. 


### Running Mist with webui

Use the following command to boost webui. See the [quickstart document](https://mist-documentation.readthedocs.io/en/latest/content/quickstart.html) for more information.
```
python mist-webui.py
```


### Robustness under input transformation

We use the following scripts to crop and resize the Misted images to test the effects of Mist under input transformation. 
```
python utils/postprocess.py
```


## Validation

Refer to the [validation document](https://mist-documentation.readthedocs.io/en/latest/content/validation.html) to validate the effects of Mist.
