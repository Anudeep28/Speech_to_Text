# Let us now functionize the model selection
import torchvision
import torch
from torch import nn

def create_model(model_name: str,
                 out_features: int=3,
                 seed:int=42,
                 device: str="cuda" if torch.cuda.is_available() else "cpu"):
    """Creates an EfficientNetB2/ EffnetV2_s / ViT_B_16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): feature extractor model. 
        transforms (torchvision.transforms): respective model image transforms.
    """
    assert model_name in "effnetb2" or model_name == "effnetv2_s" or model_name == "vit_b_16", "Model name should be effnetb2 or effnetv2_s or vit_b_16"
    if model_name == "effnetb2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        transforms = weights.transforms()
        model = torchvision.models.efficientnet_b2(weights=weights).to(device)
        dropout = 0.3
        in_features = 1408
    elif model_name == "effnetv2_s":
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        transforms = weights.transforms()
        model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
        dropout = 0.2
        in_features = 1280
    elif model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        transforms = weights.transforms()
        model = torchvision.models.vit_b_16(weights=weights).to(device)
        #dropout = 0.2
        in_features = 768

    # Freeze the base layer of the models
    for param in model.parameters():
        param.requires_grad = False

    # Update the classifier head
    if model_name != "vit_b_16":
        torch.manual_seed(seed)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features,out_features=out_features)
        ).to(device)
    elif model_name == "vit_b_16":
        torch.manual_seed(seed)
        model.heads = nn.Sequential(nn.Linear(in_features=768, # keep this the same as original model
                                          out_features=out_features)).to(device) # update to reflect target number of classes

    # set the model name
    model.name = model_name
    print(f"[INFO] Creating {model_name} feature extractor model...")
    return model, transforms
