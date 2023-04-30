import argparse
import PIL
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Configs for Mist V1.2")
    parser.add_argument(
        "-img",
        "--input_image_path",
        type=str,
        default="test/sample.png",
        help="path of input image",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="misted_sample",
        help="path of saved image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vangogh",
        help="path of output dir",
    )
    parser.add_argument(
        "-inp",
        "--input_dir_path",
        type=str,
        default=None,
        help="Path of the dir of images to be processed.",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=int,
        default=16,
        help=(
            "The strength of Mist"
        ),
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100,
        help=(
            "The step of Mist"
        ),
    )
    parser.add_argument(
        "-in_size",
        "--input_size",
        type=int,
        default=512,
        help=(
            "The input_size of Mist"
        ),
    )
    parser.add_argument(
        "-b",
        "--block_num",
        type=int,
        default=1,
        help=(
            "The number of partitioned blocks"
        ),
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        help=(
            "The mode of MIST."
        ),
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=1,
        help=(
            "The fused weight under the fused mode."
        ),
    )
    parser.add_argument(
        "--mask",
        default=False,
        action="store_true",
        help=(
            "Whether to mask certain region of Mist or not. Work only when input_dir_path is None. "
        ),
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="test/processed_mask.png",
        help="Path of the mask.",
    )
    parser.add_argument(
        "--non_resize",
        default=False,
        action="store_true",
        help=(
            "Whether to keep the original shape of the image or not."
        ),
    )
    args = parser.parse_args()
    return args


def load_mask(mask):
    mask = np.array(mask)[:,:,0:3]
    for p in range(mask.shape[0]):
        for q in range(mask.shape[1]):
            # if np.sum(mask[p][q]) != 0:
            #     mask[p][q] = 255
            if mask[p][q][0] != 255:
                mask[p][q] = 0
            else:
                mask[p][q] = 255
    return mask


def closing_resize(image_path: str, input_size: int, block_num: int = 1, no_load: bool = False) -> PIL.Image.Image:
    if no_load:
        im = image_path
    else:
        im = Image.open(image_path)
    target_size = list(im.size)

    resize_parameter = min(target_size[0], target_size[1])/input_size
    block_size = 8 * block_num 
    target_size[0] = int(target_size[0] / resize_parameter) // block_size * block_size
    target_size[1] = int(target_size[1] / resize_parameter) // block_size * block_size
    img = im.resize(target_size)
    return img, target_size


def load_image_from_path(image_path: str, input_width: int, input_height: int = 0, no_load: bool = False) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    if input_height == 0:
        input_height = input_width
    if no_load:
        img = image_path.resize((input_width, input_height),
                                            resample=PIL.Image.BICUBIC)
    else:
        img = Image.open(image_path).resize((input_width, input_height),
                                            resample=PIL.Image.BICUBIC)
    return img

