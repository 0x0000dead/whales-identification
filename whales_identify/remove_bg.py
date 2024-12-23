# будем использовать готовую либу
# под копотом используется U2net обученная
from rembg import remove
from PIL import Image

input_path = "whale_image.jpg"
output_path = "result.png"

with open(input_path, "rb") as inp_file:
    input_image = inp_file.read()

output_image = remove(input_image)

with open(output_path, "wb") as out_file:
    out_file.write(output_image)

print(f"Background removed. Saved as {output_path}")
