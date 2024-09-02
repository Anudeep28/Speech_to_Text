# # Get all test data paths
# from PIL import Image
# from tqdm import tqdm
# from pathlib import Path
# # test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
# # test_labels = [path.parent.stem for path in test_data_paths]

# # Create a function to return a list of dictionaries with sample, label, prediction, pred prob
# def pred_and_store(test_paths, model, transform, class_names, device):
#   # if not test_paths:
#   #   assert test_dir, "Please give a test_dir or test_paths"
#   #   test_paths = list(Path(test_dir).glob("*/*.jpg"))
#   # To recover the ckass names if any fro the dir
#   #test_labels = [path.parent.stem for path in test_data_paths]
#   test_pred_list = []
#   for path in tqdm(test_paths):
#     # Create empty dict to store info for each sample
#     pred_dict = {}

#     # Get sample path
#     pred_dict["image_path"] = path

#     # Get class name
#     class_name = path.parent.stem
#     pred_dict["class_name"] = class_name

#     # Get prediction and prediction probability
#     # from PIL import Image
#     img = Image.open(path) # open image
#     transformed_image = transform(img).unsqueeze(0) # transform image and add batch dimension
#     model.eval()
#     with torch.inference_mode():
#       pred_logit = model(transformed_image.to(device))
#       pred_prob = torch.softmax(pred_logit, dim=1)
#       pred_label = torch.argmax(pred_prob, dim=1)
#       pred_class = class_names[pred_label.cpu()]

#       # Make sure things in the dictionary are back on the CPU 
#       pred_dict["pred_prob"] = pred_prob.unsqueeze(0).max().cpu().item()
#       pred_dict["pred_class"] = pred_class
  
#     # Does the pred match the true label?
#     pred_dict["correct"] = class_name == pred_class

#     # print(pred_dict)
#     # Add the dictionary to the list of preds
#     test_pred_list.append(pred_dict)

#   return test_pred_list

import pathlib
import torch
import torchvision
from PIL import Image
from timeit import default_timer as timer 
from tqdm.auto import tqdm
from typing import List, Dict

# 1. Create a function to return a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time
def pred_and_store(paths: List[pathlib.Path], 
                   model: torch.nn.Module,
                   transform: torchvision.transforms,  # type: ignore
                   class_names: List[str], 
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
    
    # 2. Create an empty list to store prediction dictionaires
    pred_list = []
    
    # 3. Loop through target paths
    for path in tqdm(paths):
        
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name
        
        # 6. Start the prediction timer
        start_time = timer()
        
        # 7. Open image path
        img = Image.open(path)
        
        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device) 
        
        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model = model.to(device)
        model.eval()
        
        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image) # perform inference on target sample 
            pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
            pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
            pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class
            
            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)
    
    # 15. Return list of prediction dictionaries
    return pred_list