# Mist-webui-v1.0

Mist is a project aims at x. See [documents](https:) for more information.


## TODO

## Setup

Our code builds on, and shares most requirements with  [stable-diffusion](https://github.com/CompVis/stable-diffusion). In addition, [advertorch0.2.4](https://github.com/BorealisAI/advertorch) is required for performancing the attack. To set up the environment, please run: 

```
conda env create -f environments.yml
conda activate mist
```

In some cases the Pillow may not work properly, resulting in a more visible noise. Try reinstall Pillow if you find the $l_{infinity}$ distance between the misted image and the original image greatly larger the given strength.


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

Refer to the [documents](https:) to validate the effects of mist.
