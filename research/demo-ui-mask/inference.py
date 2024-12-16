from tensorflow import keras
from PIL import Image
import os
from numpy import asarray
import numpy as np
IMG_SHAPE = 224
model = keras.models.load_model(r'./models/vgg19_075.pth')

a = dict({'1': 0, '10': 1, '100': 2, '101': 3, '102': 4, '11': 5, '12': 6, '13': 7, '14': 8, '15': 9, '16': 10, '17': 11, '18': 12, '19': 13, '2': 14, '20': 15, '21': 16, '22': 17, '23': 18, '24': 19, '25': 20, '26': 21, '27': 22, '28': 23, '29': 24, '3': 25, '30': 26, '31': 27, '32': 28, '33': 29, '34': 30, '35': 31, '36': 32, '37': 33, '38': 34, '39': 35, '4': 36, '40': 37, '41': 38, '42': 39, '43': 40, '44': 41, '45': 42, '46': 43, '47': 44, '48': 45, '49': 46, '5': 47, '50': 48, '51': 49, '52': 50,
         '53': 51, '54': 52, '55': 53, '56': 54, '57': 55, '58': 56, '59': 57, '6': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '7': 69, '70': 70, '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '8': 80, '80': 81, '81': 82, '82': 83, '83': 84, '84': 85, '85': 86, '86': 87, '87': 88, '88': 89, '89': 90, '9': 91, '90': 92, '91': 93, '92': 94, '93': 95, '94': 96, '95': 97, '96': 98, '97': 99, '98': 100, '99': 101})
b = {v: k for k, v in a.items()}


def getBestFive(raw_data_path):
    for whale in os.listdir(raw_data_path):
        filename_list = [name.split('.')[0]
                         for name in os.listdir(raw_data_path + whale)]
        sorted_filename_list = sorted(list(set(filename_list)))
        print(whale)
        for filename in sorted_filename_list:
            pic1 = Image.open(raw_data_path + whale + '/' + filename + '.jpg')
            pic2 = Image.open(raw_data_path + whale + '/' + filename + '.png')
            black = Image.new('RGB', pic1.size, (255, 255, 255))

            pic = Image.composite(pic1, black, pic2)
            resized_pic = pic.resize((IMG_SHAPE, IMG_SHAPE))
            scalded_pic = asarray(resized_pic)/255
            scalded_pic = np.expand_dims(scalded_pic, axis=0)
            df = model.predict(scalded_pic)
            df = df[0].argsort()[-5:][::-1]
            print([b.get(k) for k in df])
            break
