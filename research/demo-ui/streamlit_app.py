import heapq
import math
import numpy as np
import pandas
import streamlit as st
from PIL import Image
from inference_transformers import create_loader, inference_nn
import cv2
import pandas as pd
from collections import Counter


@st.cache_data()
def load_data():
    return Image.open(r'./resources/defaultPhoto.jpg')


db = pd.read_csv(r'./resources/database.csv')


def use_network(img):
    test_loader = create_loader(img)
    tags, prob = inference_nn(test_loader)
    animals = []
    for i in tags:
        animals.append(db[db['individual_id'] == i]['species'].mode()[0])

    return [animals, prob, tags]


# top_ls - list of tuples
def convert_to_prob(top_ls):
    if len(top_ls) == 0:
        return []

    probs = [math.exp(elem[1]) for elem in top_ls]
    sigma = sum(probs)

    return [(top_ls[i][0] + 1, probs[i] / sigma) for i in range(len(probs))]


# add indices
def add_first_col(ls):
    return [(i + 1, *ls[i]) for i in range(len(ls))]


def main():
    st.title("Идентификация морских млекопитающих со снимков аэрофотосъемки")
    file_photo = st.file_uploader("Загрузить данные:", type=['jpg'])

    # load default image
    if file_photo is not None:
        photo = Image.open(file_photo)
    else:
        photo = load_data()

    # show images
    st.header("Фото млекопитающего:")
    st.image(photo, width=300, use_column_width='always')

    animals, probs, tags = use_network(list(np.array(photo)))
    print(animals, tags, probs)

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.subheader(Counter(animals).most_common(1)[0][0])
    st.table(pandas.DataFrame({'individual ID': tags, 'Animals': animals, 'Prob': [str(f'{round(i*1000, 2)}%') for i in probs]},
                              index=[i for i in range(len(tags))]))
    st.caption('при поддержке Фонда содействия инновациям')


if __name__ == '__main__':
    main()
