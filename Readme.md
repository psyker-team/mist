# Mist-webui-v1.0

Mist is a project aims at x. See [documents](https:) for more information.


## TODO

## Setup

Our code builds on, and shares most requirements with  [stable-diffusion](https://github.com/CompVis/stable-diffusion). In addition, [advertorch0.2.4](https://github.com/BorealisAI/advertorch) is required for performancing the attack. To set up the environment, please run: 

```
conda env create -f environment.yaml
conda activate mist
```

Official Stable-diffusion-model v1.4 checkpoint is also required, available in [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt).

Currently, the model can be downloaded by:
```
wget -O https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
os.rename sd-v1-4.ckpt
rename sd-v1-4.ckpt model.ckpt
mv model.ckpt models/ldm/stable-diffusion-v1
```

## Usage

### Scripts

Use the following command to generate misted image for test/man.png:
```
python mist_v2.py 16 100 512 1 2 1
```
The misted image is saved in test/misted_man_16_100_512_1_2_1.png


### Webui

Use the following command to generate webui. See [documents](https:) for more information.
```
python mist-webui.py
```
