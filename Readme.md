# Mist

Updated: Good news! Our paper is acceped by ICML 2023 as Oral presentation. The paper can be found at [arxiv](https://arxiv.org/abs/2302.04578) currently. Mist is based upon the paper with some extensions (See our [technical report](http://arxiv.org/abs/2305.12683) for more details).


Mist is a powerful image preprocessing tool designed for the purpose of protecting the style and content of images from being mimicked by state-of-the-art AI-for-Art applications, including Stable Diffusion, NovelAI, and scenario.gg. By adding watermarks to the images, Mist renders them unrecognizable for the models employed by AI-for-Art applications. Attempts by AI-for-Art applications to mimic these Misted images will be ineffective, and the output image of such mimicry will be scrambled and unusable as artwork. For more details on Mist, refer to our [documentation](https://mist-documentation.readthedocs.io/en/latest) and [homepage](https://psyker-team.github.io/).

This repository provides the complete source code of Mist. Source code can be used to build Mist-WebUI from scratch or deploy a Mist remote service on a server. 



## Setup

Our code builds on, and shares most requirements with [stable-diffusion](https://github.com/CompVis/stable-diffusion). In addition, [advertorch0.2.4](https://github.com/BorealisAI/advertorch) is required for performing the attack. To set up the environment, please run: 

```
conda env create -f environments.yml
conda activate mist
pip install --force-reinstall pillow
```

Note that PyPI appears to install an incomplete kit of Pillow thus a reinstallation is in need. We highly recommend reinstalling Pillow after you activate the virtual environment mist for the first time.


Official Stable-diffusion-model v1.4 checkpoint is also required, available at [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt). Currently, the model can be downloaded by running following commands:

```
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p  models/ldm/stable-diffusion-v1
mv sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

## Usage

### Running Mist with Python scripts

`mist_v2.py` takes the image stored in `test/sample.png` as the input and outputs the watermarked image also in `test/`. This script has 6 parameters.

| Parameter       | Value range (Recommended)               | Note                                                         |
| --------------- | --------------------------------------- | ------------------------------------------------------------ |
| Strength        | [1, 32]                                 | The strength of the watermark.                               |
| Steps           | [1, 1000]                               | The number of steps to optimize the watermark.               |
| Output size     | {256, 512, 768}                         | The size of the output image, square only.                   |
| Blocking Number | {1(1x1), 2 (2x2)}                       | If true, separate the image into blocks of BxB and add watermarks to these blocks, respectively. |
| Mode            | {0 (textural), 1 (semantic), 2 (fused)} | Watermark mode. See documentation for details.               |
| Fused weight    | [1, 5]                                  | Balance the weight of textural mode and semantic mode in fused mode. |

The parameters must be provided in sequence as mentioned in the table. For example, use the following command to Mist the image with Strength 16, Steps 100, output size 512, blocking number 1, mode 2 (fused mode), and fused weight 1:

```
python mist_v2.py 16 100 512 1 2 1
```

Users can also mist images in a directory. For example, use the following command to generate the misted image in `test/vangogh`:
```
python mist_v2_vangogh.py 16 100 512 1 2 1
```

### Running Mist with WebUI

Use the following command to boost webui. See the [quickstart document](https://mist-documentation.readthedocs.io/en/latest/content/quickstart.html) for more information.
```
python mist-webui.py
```


### Robustness under input transformation

We provide scripts to crop and resize the Misted images to evaluate the robustness of Mist under input transformation. See the script `utils/postprocess.py`.


## Validation

Refer to the [validation document](https://mist-documentation.readthedocs.io/en/latest/content/validation.html) to validate the effects of Mist.


## License

This project is licensed under the [GPL-3.0 license](https://github.com/mist-project/mist/blob/main/LICENSE). 


Part of the code is based on the [stable-diffusion](https://github.com/CompVis/stable-diffusion). You can find the [license](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE) of stable-diffusion in their repository. 

It is notable that Mist requires an open gradient flow of stable diffusion model for end-to-end adversarial perturbation. Thus following files provided by stable-diffusion are modified in support of an open gradient flow:

```
models/diffusion/ddpm.py  ---->  models/diffusion/ddpmAttack.py 
configs/stable-diffusion/v1-inference.yaml  ---->  configs/stable-diffusion/v1-inference-attack.yaml
```
 
## Citation
If you find our work valuable and utilize it, we kindly request that you cite our paper.

```
@inproceedings{liang2023adversarial,
  title={Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples},
  author={Liang, Chumeng and Wu, Xiaoyu and Hua, Yang and Zhang, Jiaru and Xue, Yiming and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={International Conference on Machine Learning},
  pages={20763--20786},
  year={2023},
  organization={PMLR}
}
```
```
@article{liang2023mist,
  title={Mist: Towards Improved Adversarial Examples for Diffusion Models},
  author={Liang, Chumeng and Wu, Xiaoyu},
  journal={arXiv preprint arXiv:2305.12683},
  year={2023}
}
```
