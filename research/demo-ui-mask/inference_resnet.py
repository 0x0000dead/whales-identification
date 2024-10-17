import cv2
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import glob
from PIL import Image

device = torch.device("cuda:0")
import numpy as np

def get_prediction_resnet(img, model):
    '''
    :param img:  image
    :param model_path: path to model
    :return: Top 5 predictions for each image
    '''
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
    img = cv2.resize(img, (256, 256))

    trf = albu.Compose([albu.Lambda(image=preprocessing_fn)])
    img = trf(image=img)['image']
    img = img.transpose(2, 0, 1).astype('float32')

    x_tensor = torch.from_numpy(img).to("cuda").unsqueeze(0)

    out = model(x_tensor)
    top_5 = torch.topk(out, 5)
    top_5_arg = top_5.indices.cpu().numpy()[0]
    return top_5_arg


import os


def form_result_to_submission(raw_data_path):
    model = torch.load(r"C:\Users\Sergey\Desktop\MODELS\resnet50\5_0.4741_0.3084.pth")
    model.eval()
    for whale in os.listdir(raw_data_path):
        res = dict()
        print(whale)
        for whale_inner_folder in os.listdir(raw_data_path + whale):
            filename_list = [name.split('.')[0] for name in os.listdir(raw_data_path + whale + '/' + whale_inner_folder)]
            sorted_filename_list = sorted(list(set(filename_list)))
            for filename in sorted_filename_list:
                pic1 = Image.open(raw_data_path + whale + '/' + whale_inner_folder + '/' + filename + '.jpg')
                pic2 = Image.open(raw_data_path + whale + '/' + whale_inner_folder + '/' + filename + '.png')
                black = Image.new('RGB', pic1.size, (255, 255, 255))
                pic = Image.composite(pic1, black, pic2)
                out = get_prediction_resnet(np.array(pic), model)
                for i in range(len(out)):
                    res[out[i] + 1] = res.get(out[i] + 1, 0) + 1
        sorted_di = sorted(res.items(), key=lambda x: x[1], reverse=True)
        print(sorted_di)



if __name__ == '__main__':
    form_result_to_submission('C:/Users/Sergey/Downloads/Whale_ReId_test_mm/')
