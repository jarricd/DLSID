import torch
import argparse
import pathlib
import sys
import cv2
import os
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import img2tensor, tensor2img
import torch.nn.functional as F
import json
import logging

logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ])

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Run inference/test on images.", epilog="This program does not have supercow powers.")
    argument_parser.add_argument("input_path", help="Input images for test.")
    argument_parser.add_argument("output_path", help="Output directory.")
    argument_parser.add_argument("--weights", help="Path to experiment weights directory")
    argument_parser.add_argument("--json", help="Scores json file")
    argument_parser.add_argument("--file_count", help="File count to be restored.", type=int, default=10)
    args = argument_parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    output_path = pathlib.Path(args.output_path)
    weights_path = pathlib.Path(args.weights)
    json_path = pathlib.Path(args.json)

    if not input_path.is_dir():
        sys.exit(1)
    
    if not output_path.is_dir():
        sys.exit(2)

    if not weights_path.is_dir():
        sys.exit(3)

    if not json_path.is_file():
        sys.exit(4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores_json = json.loads(json_path.read_text())
    weight_files = list(weights_path.rglob("*pth"))
    input_imgs: list = list(input_path.glob("*jpg"))
    # input_imgs.extend(list(input_path.glob("*png")))
    curr_count = 0
    for weight_file in weight_files:
        logging.info(f"Using weights: {weight_file}")
        net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False).to(device)
        if str(device) == "cpu":
            checkpoint = torch.load(weight_file, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(weight_file)
        net.load_state_dict(checkpoint['params'])        
        net.eval()
        try:
            os.mkdir(f"{output_path}/{weight_file.stem}")
        except FileExistsError as e:
            logging.warning("Directory exists, continuing.")

        # oh god please forgive me for this mess
        exposure = 0
        for img_num, input_img in enumerate(input_imgs):
            params = scores_json[f"{input_img.stem}{input_img.suffix}"]
            exposure = params["degradations"]["exposure_decrease"]    
            logging.debug   (f"File {input_img} score is: {exposure}")
            if exposure < 0.4:
                logging.info(f"Processing file: {input_img}")
                tested_img = cv2.imread(str(input_img))
                img_tensor = img2tensor(tested_img / 255, bgr2rgb=True, float32=True)
                img_tensor = img_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    H, W = img_tensor.shape[2:]
                    # image from the original code
                    img_tensor = check_image_size(img_tensor, 8)
                    output_t = net(img_tensor)
                    output_t = output_t[:,:,:H,:W]
                    output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))

                torch.cuda.empty_cache()
                output = output.astype('uint8')
                logging.info(f"{output_path}/{weight_file.stem}/{input_img.stem}{input_img.suffix}")
                cv2.imwrite(f"{output_path}/{weight_file.stem}/{input_img.stem}{input_img.suffix}", output)
                if curr_count > args.file_count:
                    curr_count = 0
                    break
                curr_count += 1
