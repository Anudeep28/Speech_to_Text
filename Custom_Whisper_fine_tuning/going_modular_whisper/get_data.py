# import os
# import zipfile

# from pathlib import Path

# import requests

# # Setup path to data folder
# data_path = Path("data/")
# image_path = data_path / "pizza_steak_sushi"

# # If the image folder doesn't exist, download it and prepare it... 
# if image_path.is_dir():
#     print(f"{image_path} directory exists.")
# else:
#     print(f"Did not find {image_path} directory, creating one...")
#     image_path.mkdir(parents=True, exist_ok=True)
    
# # Download pizza, steak, sushi data
# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#     request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#     print("Downloading pizza, steak, sushi data...")
#     f.write(request.content)

# # Unzip pizza, steak, sushi data
# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#     print("Unzipping pizza, steak, sushi data...") 
#     zip_ref.extractall(image_path)

# # Remove zip file
# os.remove(data_path / "pizza_steak_sushi.zip")

# # # Setup train and testing paths
# # train_dir = image_path / "train"
# # test_dir = image_path / "test"

# # train_dir, test_dir

import os

# Function to load audio files and their transcripts
def load_data(audio_dir, transcript_dir):
    data = []
    #print("path: ",os.getcwd())
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(audio_dir, filename)
            transcript_path = os.path.join(transcript_dir, filename.replace(".wav", ".txt"))
            
            with open(transcript_path, 'r') as f:
                transcript = f.read().strip()
            
            data.append({"audio": audio_path, "text": transcript})
    return data
