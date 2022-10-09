import numpy as np

from PIL import Image
import cv2
from mmseg.apis import inference_segmentor, init_segmentor
from util.img_util import convert_PIL_to_base64, convert_image_to_ABImage


# 添加颜色 flag = True 添加颜色 flag = false 不添加颜色去除背景保留晾制烟叶区域
def apply_mask(image, mask, palette, alpha=0.5):
    img_ann = image.copy() # 类别标注
    img_seg = image.copy() # 去除背景保留晾制烟叶区域
    for c in range(3):
        for l in range(1, len(palette)):
            img_ann[:, :, c] = np.where((mask==l), image[:, :, c] * (1 - alpha) + alpha * palette[l][c], img_ann[:, :, c])
            
        img_seg[:, :, c] = np.where(mask==0, 0, image[:, :, c])
    return img_ann, img_seg

def gy_ratio(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tobacco_area = np.count_nonzero(img_gray > 0)
    bg_area = img_gray.shape[0] * img_gray.shape[1] - tobacco_area

    img_luv = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
    L, U, V = cv2.split(img_luv)
    
    yellow_thresh = 96
    y_area = np.count_nonzero(U > yellow_thresh)
    g_area = np.count_nonzero(U <= yellow_thresh)

    g_ratio = (g_area - bg_area) / tobacco_area
    y_ratio = (y_area) / tobacco_area

    return g_ratio, y_ratio

def infer_cal(model_path, config_path, img_path):
    # config_path = 'configs\\swin_tobacco.py'
    # checkpoint_path = 'checkpoints\\swin_tobacco_30000.pth'
    
    DEVICE = 'cpu'
    PALETTE = [[128, 128, 128], [25, 255, 25], [255, 25, 25], [129, 127, 38], [120, 69, 125], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_path, model_path, device=DEVICE)

    img = Image.open(img_path)
    img_masked = np.array(img).copy()

    result = inference_segmentor(model, img_path)[0]

    _, img_seg = apply_mask(img_masked, result, palette=PALETTE)
    g_ratio, y_ratio = gy_ratio(img_seg)

    img_seg_abi = convert_image_to_ABImage(Image.fromarray(img_seg)) # 与grpc proto image定义对象保持一致
    #img_seg_base64 = convert_PIL_to_base64(Image.fromarray(img_seg))

    return (g_ratio, y_ratio, img_seg_abi)

def main(model_path, config_path, img_path):
    return infer_cal(model_path, config_path, img_path)
    
# if __name__ == '__main__':
#     main()
