from pathlib import Path
import requests
# get an image of pizza/steak/sushi
# get an image of not pizza/steak/sushi
def download_image(data_dir:Path, image_name_ext: str="default.jpg", image_url: str=""):
    assert image_url != "", "Please provide a valid url"
    with open(data_dir / image_name_ext, "wb") as f:
        request = requests.get(image_url)
        #print("Downloading pizza, steak, sushi data...")
        f.write(request.content)
