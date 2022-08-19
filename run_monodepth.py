import os
import torch
import cv2
import argparse
import util.io
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


models = {
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
    "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
}


def run(input_path, output_dir, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): input filepath
        output_dir (str): output directory
        model_type (str): type of model (dpt_large | dpt_large, dpt_hybrid, dpt_hybrid_kitti, dpt_hybrid_nyu)
        optimize (bool):
    """
    print("Initializing...")

    print(f'Using model {model_type}')
    model_path = f'{os.path.dirname(__file__)}/{models[model_type]}'

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # load network
    if model_type == "dpt_large":
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352
        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480
        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError(f"model_type '{model_type}' not implemented, see help for parameter --model_type")

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    if os.path.isdir(input_path):
        for image_path in os.listdir(input_path):
            if '.jpg' in image_path or '.png' in image_path:
                prediction = predict_depth(os.path.join(input_path, image_path), device, transform, model, model_type, output_dir, optimize)

                output_filename = output_dir + '/' + os.path.basename(image_path).split('.')[0] + '_depth'
                print(f'Saving to {output_filename}')
                util.io.write_depth(output_filename, prediction, bits=2, absolute_depth=args.absolute_depth)

    else:
        prediction = predict_depth(input_path, device, transform, model, model_type, output_dir, optimize)

        output_filename = output_dir + '/' + os.path.basename(input_path).split('.')[0] + '_depth'
        print(f'Saving to {output_filename}')
        util.io.write_depth(output_filename, prediction, bits=2, absolute_depth=args.absolute_depth)

    print("finished")


def predict_depth(filepath: str, device, transform, model, model_type, output_dir, optimize: bool):
    print(f'Processing {filepath}')

    img = util.io.read_image(filepath)

    if args.kitti_crop is True:
        height, width, _ = img.shape
        top = height - 352
        left = (width - 1216) // 2
        img = img[top: top + 352, left: left + 1216, :]

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        if model_type == "dpt_hybrid_kitti":
            prediction *= 256

        if model_type == "dpt_hybrid_nyu":
            prediction *= 1000.0

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input", help="input image / folder"
    )

    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="folder for output image, if not provided, will save next to input image",
    )

    parser.add_argument(
        "-t", "--model_type",
        default="dpt_large",
        help="model type [dpt_large | dpt_large, dpt_hybrid, dpt_hybrid_kitti, dpt_hybrid_nyu]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    # prepare output folder
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.input)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input,
        args.output_dir,
        args.model_type,
        args.optimize,
    )
