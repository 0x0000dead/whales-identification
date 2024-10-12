from inference_transformers import create_loader, inference_nn
import cv2
import pandas as pd

    
image = list(cv2.imread(r'resources\defaultPhoto.jpg'))
test_loader = create_loader(image)
tags, prob = inference_nn(test_loader)
db = pd.read_csv(r'resources/database.csv')

animals = []
for i in tags:
        animals.append(db[db['individual_id'] == i]['species'].mode())
popular_animal = db[db['individual_id'] == tags[0]]['species'].mode()
x = pd.DataFrame({'ID': tags, 'Prob': prob}, index=False)

print(popular_animal)