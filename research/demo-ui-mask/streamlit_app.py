import heapq
import math
import numpy as np
import pandas
import streamlit as st
from PIL import Image

MODEL_PATH = "./6_0.4796_0.3058.pth"


@st.cache(suppress_st_warning=True)
def load_data():
    return Image.open('./resources/defaultPhoto.jpg'), Image.open('./resources/defaultMask.png')


def use_network(img, model):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        'resnet50', 'imagenet')
    # resize 256 256
    img = cv2.resize(img, (256, 256))
    trf = albu.Compose([albu.Lambda(image=preprocessing_fn)])
    img = trf(image=img)['image']
    img = img.transpose(2, 0, 1).astype('float32')
    x_tensor = torch.from_numpy(img).to("cuda").unsqueeze(0)
    out = model(x_tensor)
    # top_2 = torch.topk(out, 5)
    # top_2_arg = top_2.indices.cpu().numpy()[0]
    return out[0][1:]


def find_top(predict_list, top_size):
    ls = [(i, predict_list[i]) for i in range(len(predict_list))]
    return heapq.nlargest(top_size, ls, key=lambda item: item[1])


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
    st.title("Фонд содействий инновациям.")
    file_photo = st.file_uploader("Загрузите фото кита:", type=['jpg'])
    file_mask = st.file_uploader("Загрузите маску:", type=['png'])

    top_size = st.number_input(
        'Размер ТОПа:', min_value=1, max_value=50, value=5)

    # load default image
    if file_photo is not None and file_mask is not None:
        photo = Image.open(file_photo)
        mask = Image.open(file_mask)
    else:
        photo, mask = load_data()

    mask = mask.resize(photo.size)

    # show images
    col1, col2 = st.columns(2)
    with col1:
        st.text("Фото кита:")
        st.image(photo)
    with col2:
        st.text("Маска:")
        # multiply img with mask with pil

        photo_with_mask = Image.composite(photo, Image.new(
            "RGB", photo.size, (255, 255, 255)), mask)
        st.image(photo_with_mask)

    photo_with_mask = Image.composite(photo, Image.new(
        "RGB", photo.size, (255, 255, 255)), mask)
    photo_arr, mask_arr = np.array(photo), np.array(mask)
    predict_list = use_network(np.array(photo_with_mask), model)

    # find top 5 with indexes
    top_ls = find_top(predict_list, top_size)

    # convert to probability
    top_ls = convert_to_prob(top_ls)
    top_ls = add_first_col(top_ls)

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(pandas.DataFrame(top_ls, columns=['Number', 'Id', 'Probability']))


if __name__ == '__main__':
    main()
