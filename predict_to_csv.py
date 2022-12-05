import argparse

from mmseg.apis import inference_segmentor, init_segmentor
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

config_file = 'configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py'
checkpoint_file = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'

def get_class_freq(ndarray):
    freq = [0] * 19
    unique, counts = np.unique(ndarray, return_counts=True)    
    for i in range(len(counts)):
      freq[unique[i]] = counts[i]/ndarray.size*100
    return freq

def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentor')
    parser.add_argument('--config', help='config path')
    parser.add_argument('--checkpoint', help='checkpoint path')
    parser.add_argument('--path', help='dataset path')
    parser.add_argument('--chunk', type=int, help='chunk')
    args = parser.parse_args()
    return args

def get_image_path(image_path, chunk):
    image_list = []
    for i in range(chunk*20000, (chunk+1) * 20000):
        if(os.path.exists(f"{image_path}/{i}.png")):
            image_list.append(f"{image_path}/{i}.png")
    return image_list


def main():
    args = parse_args()
    print(args.path, args.chunk)

    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')
    class_def = list(model.CLASSES)
    df = pd.DataFrame(columns=["image_id"]+class_def)
    image_list = get_image_path(args.path, args.chunk)
    for i in tqdm(range(len(image_list))):
        img = image_list[i]
        result = inference_segmentor(model, img)
        image_id = img.split("/")[-1][:-4]
        pred_list = get_class_freq(result[0])
        df.loc[len(df)] = [str(image_id)] + pred_list
    if(not os.path.exists("./output")):
        os.mkdir("./output")
    df.to_csv(f"./output/result_{args.chunk}.csv", index=False)

if __name__ == '__main__':
    main()
    