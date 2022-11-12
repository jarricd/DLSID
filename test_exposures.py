import degrad.handcrafted_degradations
import torch
import cv2
from basicsr.archs.zerodce_arch import ConditionZeroDCE

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = "LEDNet/weights/ce_zerodce.pth"
    net = ConditionZeroDCE().to(device)
    if str(device) == "cpu":
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint)
    net.eval()

    img_path = "/media/jetson/JETSON/X1/00173.png"
    input_img = cv2.imread(img_path)
    degraded_img = degrad.handcrafted_degradations.change_brightness(input_img, 0.5)
    degraded_img_dce = degrad.handcrafted_degradations.zero_dce_exposure(net, degraded_img, 0.5, device)
    cv2.imwrite("00173_hand.png", degraded_img)
    cv2.imwrite("00173_dce.png", degraded_img_dce)

