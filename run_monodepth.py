import os
import torch
import cv2
import argparse
import util.io
from torchvision.transforms import Compose
from attrs import define, field

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


models = {
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
    "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
}


@define
class DepthMap:
    input_filepath: str = field()
    prediction = field()
    model_type: str = field()
    absolute: bool = field()

    def __mul__(self, other: 'DepthMap'):
        new_filepath = os.path.join(
            os.path.dirname(self.input_filepath),
            os.path.splitext(os.path.basename(self.input_filepath))[0] + '_x_' + os.path.splitext(os.path.basename(other.input_filepath))[0] +  os.path.splitext(os.path.basename(other.input_filepath))[1]
        )
        self.model_type = self.model_type if other.model_type == self.model_type else f'{self.model_type}_x_{other.model_type}'
        return DepthMap(new_filepath, self.prediction * other.prediction)

    def __rmul__(self, other: 'DepthMap'):
        return self.__mul__(other)

    def save(self, output_dir: str):
        output_filename = output_dir + '/' + os.path.basename(self.input_filepath).split('.')[0] + '_depth_' + self.model_type
        print(f'Saving to {output_filename}')
        util.io.write_depth(output_filename, self.prediction, bits=2, absolute_depth=self.absolute)


@define
class DPTMonoDepth:
    """
    model_type (str): type of model (dpt_large | dpt_large, dpt_hybrid, dpt_hybrid_kitti, dpt_hybrid_nyu)
    optimize (bool):
    """
    model_type: str = field(default='dpt_hybrid')
    optimize: bool = field(default=True)
    absolute_depth = field(default=True)

    device = field(init=False)
    model = field(init=False)
    transform = field(init=False)

    def __attrs_post_init__(self):
        print("Initializing...")

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: %s" % self.device)

        self.model, self.transform = self._prepare_model(self.model_type)

    def _prepare_model(self, model_type) -> tuple:
        print(f'Preparing model {model_type}')
        model_path = f'{os.path.dirname(__file__)}/{models[model_type]}'

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
        elif self.model_type == "dpt_hybrid_nyu":
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
            raise ValueError(f"model_type '{self.model_type}' not implemented, see help for parameter --model_type")

        model.eval()

        if self.optimize and self.device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(self.device)

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

        return model, transform

    def predict_depths(self, input_path: str) -> list:
        """Run MonoDepthNN to compute depth maps for given path

        input_path (str): input dir or filename
        output_dir (str): output directory
        """
        result = []

        if os.path.isdir(input_path):
            for image_name in os.listdir(input_path):
                if '.jpg' in image_name or '.png' in image_name:
                    filepath = os.path.join(input_path, image_name)
                    result.append(self.__predict_single_depth(filepath))
        else:
            result.append(self.__predict_single_depth(input_path))

        return result

    def __predict_single_depth(self, filepath: str):
        print(f'Processing {filepath}')

        img = util.io.read_image(filepath)

        img_input = self.transform({"image": img})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
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

            if self.model_type == "dpt_hybrid_nyu":
                prediction *= 1000.0
                prediction *= -1

        return DepthMap(
            input_filepath=filepath,
            prediction=prediction,
            model_type=self.model_type,
            absolute=self.absolute_depth
        )


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
    depths = DPTMonoDepth(
        args.model_type,
        args.optimize,
        args.absolute_depth
    ).predict_depths(
        args.input
    )

    for depth in depths:
        depth.save(args.output_dir)

    print('Finished!')
