from os import PathLike
from pathlib import Path

import PIL.Image

from appl import Image, gen, ppl


@ppl
def query_text(image_url: str):
    "Look, it's a commercial logo."
    Image(image_url)
    "What's the text on the image? "
    "Your output should be in the format: The text on the image is: ..."
    return gen("gpt4o-mini", stop="\n")


IMAGE_URL = "https://maproom.wpenginepowered.com/wp-content/uploads/OpenAI_Logo.svg_-500x123.png"
# print(query_text(IMAGE_URL))
# The text on the image is: OpenAI


@ppl
def query_image(image_file: PathLike):
    "Which python package is this?"
    Image.from_file(image_file)
    # PIL.Image.open(image_file) # alternative, APPL recognizes an ImageFile
    return gen("gpt-4o-mini", stop="\n")


image_file = Path(__file__).parent / "pillow-logo-dark-text.webp"
print(query_image(image_file))
# Pillow
