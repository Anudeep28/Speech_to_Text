# Make a function to pred and plot Image
import torch
from PIL import Image
import matplotlib.pyplot as plt

def pred_and_plot(image_path, model, transform,
                  class_names, device: str="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts on the given image path
    Plots the image with the predicted label in it
    """
    #open image
    image = Image.open(image_path)

    # transform the image
    transformed_image = transform(image)

    # pred on image
    model.eval()
    with torch.inference_mode():
        # unsqueeze adds batch dimension to the image
        pred_logits = model(transformed_image.unsqueeze(0).to(device))
        pred_label = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)

        # plot image and pred
        plt.figure()
        plt.imshow(image)
        plt.title(f"Pred: {class_names[pred_label]}")
        plt.axis(False)
