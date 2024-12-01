from tensorflow import keras
from PIL import Image
import os
import numpy as np
from numpy import asarray
import csv

IMG_SHAPE = 224
model = keras.models.load_model('./models/vgg19_075.pth')
THRESHOLD = 0.05

a = dict({'1': 0, '10': 1, '100': 2, '101': 3, '102': 4, '11': 5, '12': 6, '13': 7, '14': 8, '15': 9, '16': 10, '17': 11, '18': 12, '19': 13, '2': 14, '20': 15, '21': 16, '22': 17, '23': 18, '24': 19, '25': 20, '26': 21, '27': 22, '28': 23, '29': 24, '3': 25, '30': 26, '31': 27, '32': 28, '33': 29, '34': 30, '35': 31, '36': 32, '37': 33, '38': 34, '39': 35, '4': 36, '40': 37, '41': 38, '42': 39, '43': 40, '44': 41, '45': 42, '46': 43, '47': 44, '48': 45, '49': 46, '5': 47, '50': 48, '51': 49, '52': 50,
         '53': 51, '54': 52, '55': 53, '56': 54, '57': 55, '58': 56, '59': 57, '6': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '7': 69, '70': 70, '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '8': 80, '80': 81, '81': 82, '82': 83, '83': 84, '84': 85, '85': 86, '86': 87, '87': 88, '88': 89, '89': 90, '9': 91, '90': 92, '91': 93, '92': 94, '93': 95, '94': 96, '95': 97, '96': 98, '97': 99, '98': 100, '99': 101})
b = {v: k for k, v in a.items()}


def getBestFive(raw_data_path):
    # OPEN CSV
    with open("users.csv", mode="w", encoding='utf-8') as w_file:
        names = ["name", "top1", "top2", "top3", "top4", "top5"]
        file_writer = csv.DictWriter(w_file, delimiter=";",
                                     lineterminator="\r", fieldnames=names)
        file_writer.writeheader()
        row_dict = dict.fromkeys(names)

        # ITERATE OVER ALL WHALE FOLDERS
        for whale in os.listdir(raw_data_path):
            guesses = dict()
            for whale_inner_folder in os.listdir(raw_data_path + whale):
                filename_list = [name.split('.')[0] for name in os.listdir(
                    raw_data_path + whale + '/' + whale_inner_folder)]
                sorted_filename_list = sorted(list(set(filename_list)))

                for filename in sorted_filename_list:
                    pic1 = Image.open(
                        raw_data_path + whale + '/' + whale_inner_folder + '/' + filename + '.jpg')
                    pic2 = Image.open(
                        raw_data_path + whale + '/' + whale_inner_folder + '/' + filename + '.png')
                    black = Image.new('RGB', pic1.size, (255, 255, 255))

                    # PREPARING IMAGE
                    pic = Image.composite(pic1, black, pic2)
                    resized_pic = pic.resize((IMG_SHAPE, IMG_SHAPE))
                    scalded_pic = asarray(resized_pic)/255
                    scalded_pic = np.expand_dims(scalded_pic, axis=0)
                    # MAKE PREDICTION
                    df = model.predict(scalded_pic)
                    cut_df = df[0].argsort()[-5:][::-1]
                    prob_cut = [df[0][k] for k in cut_df]

                    # UNKNOWN CLASS
                    if max(prob_cut) < THRESHOLD:
                        if '0' not in guesses.keys():
                            guesses['0'] = 1 - max(prob_cut)
                        else:
                            guesses['0'] += 1 - max(prob_cut)
                        continue

                    # FIVE BEST GUESSES
                    for k in cut_df:
                        if b.get(k) not in guesses.keys():
                            guesses[b.get(k)] = df[0][k]
                        else:
                            guesses[b.get(k)] += df[0][k]

            # SAVING TOP 5 GUESSES TO CSV
            best_guesses = sorted(guesses, key=guesses.get, reverse=True)[:5]
            row_dict["name"] = whale
            for i in range(5):
                row_dict['top' + str(i + 1)] = best_guesses[i]
            file_writer.writerow(row_dict)


getBestFive('./resources/')
