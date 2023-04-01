import os
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from advertorch.attacks import LinfPGDAttack


ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')


def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img


def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class target_model(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2, 
                 rate: int = 10000):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate

    def get_components(self, x):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).cuda()
        c = self.model.get_learned_conditioning(self.condition)
        loss = self.model(z, c)[0]
        return z, loss

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        zx, loss_semantic = self.get_components(x)
        zy, loss_semantic_y = self.get_components(self.target_info)
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 0:
            return - loss_semantic
        elif self.mode == 1:
            return self.fn(zx, zy)
        else:
            return self.fn(zx, zy) - loss_semantic * self.rate


def init(epsilon: int = 16, steps: int = 100, alpha: int = 1, 
         input_size: int = 512, object: bool = False, seed: int =23, 
         ckpt: str = None, base: str = None, mode: int = 2, rate: int = 10000):
    """
    Prepare the config and the model used for generating adversarial examples.
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack, 
                    the final perturbation budget will be (epsilon+1)/255.
    :param steps: Iterations of the attack.
    :param alpha: strength of the attack for each step. Measured in l_{\infinity}.
    :param input_size: Size of the input image.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused. 
                 See the document for more details about the mode.
    :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
    :returns: a dictionary containing model and config.
    """

    if ckpt is None:
        ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    seed_everything(seed)
    imagenet_templates_small_style = ['a painting']
    imagenet_templates_small_object = ['a photo']

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path).cuda()

    fn = identity_loss()

    if object:
        imagenet_templates_small = imagenet_templates_small_object
    else:
        imagenet_templates_small = imagenet_templates_small_style

    input_prompt = [imagenet_templates_small[0] for i in range(1)]
    net = target_model(model, input_prompt, mode=mode, rate=rate)
    net.eval()

    # parameter
    parameters = {
        'epsilon': epsilon/255.0 * (1-(-1)),
        'alpha': alpha/255.0 * (1-(-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'rate': rate
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]

    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]
    if tar_img is not None:
        tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
        tar_img = tar_img[:, :, :3]
    trans = transforms.Compose([transforms.ToTensor()])
    data_source = torch.zeros([1, 3, input_size, input_size]).cuda()
    data_source[0] = trans(img).cuda()
    target_info = torch.zeros([1, 3, input_size, input_size]).cuda()
    target_info[0] = trans(tar_img).cuda()
    net.target_info = target_info
    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source.shape).cuda()
    print(net(data_source, components=True))

    # Targeted PGD attack is applied.
    attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
    attack_output = attack.perturb(data_source, label)
    print(net(attack_output, components=True))

    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv


# Test the script with command: python mist_v2.py 16 100 512 1 2 1
# or the command: python mist_v2.py 16 100 512 2 2 1, which process
# the image blockwisely for lower VRAM cost

if __name__ == "__main__":
    epsilon = int(sys.argv[1])
    steps = int(sys.argv[2])
    input_size = int(sys.argv[3])
    block_num = int(sys.argv[4])
    mode = int(sys.argv[5])
    rate = 10 ** (int(sys.argv[6]) + 3)

    bls = input_size//block_num

    image_path = './test/man.png'
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
    output_name = '../test/misted_man_'
    for i in range(5):
        output_name += (sys.argv[i+1] + '_')
    if mode >= 2:
        output_name += (sys.argv[6])
    output_path = output_name + '.png'
    output.save(output_path)
