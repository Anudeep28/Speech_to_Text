import torch
from torch import nn


# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        self.patch_size = patch_size
        
        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        #print(f"Shape after patcher layer: {x_patched.shape}")
        x_flattened = self.flatten(x_patched)
        #print(f"Shape after flatten layer: {x_flattened.shape}")
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class ViT(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 num_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768,
                 dropout: float=0.1,
                 mlp_size: int=3072,
                 num_transformer_layers: int=12,
                 num_heads: int=12,
                 num_classes: int=1000):
        super().__init__()

        # Assert image size is divisible by patch size
        assert img_size % patch_size == 0, "Image size should be divisible by patch size"

        # 1. Create patch embeddings on the image
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        # 2. Create a class token
        self.class_token = nn.Parameter(torch.randn(1,1,embedding_dim, requires_grad=True))
        
        # 3. Create positional embedding
        num_patches = (img_size*img_size) // patch_size**2 # N = H*W / P^2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1,
                                                             embedding_dim))

        # 4. Create patch + positional embedding droptout
        self.embedding_dropout = nn.Dropout(p=dropout)
        # 5. Create transformer encoding layers
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embedding_dim,
        #     nhead=num_heads,
        #     dim_feedforward=mlp_size,
        #     activation="gelu",
        #     batch_first=True,
        #     norm_first=True
        # )
        # 6. Create a stack of transformer enoder
        self.transformer_encoder = nn.TransformerEncoder(
                                                            encoder_layer=nn.TransformerEncoderLayer(
                                                            d_model=embedding_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=mlp_size,
                                                            activation="gelu",
                                                            batch_first=True,
                                                            norm_first=True
                                                        ),
            num_layers=num_transformer_layers
        )
        # 7. Create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # forward method to connect all the layers above
    def forward(self,x):
        # get some dimension from x
        batch_size = x.shape[0] # [B,C,H,W]

        # Create the patch embedding
        x = self.patch_embedding(x)
        #print(x.shape)
        
        # First expand the class token across the batch size
        class_token = self.class_token.expand(batch_size, -1, -1) # "-1" means infer the dimension

        # Prepend the class token to the patch embedding
        x = torch.cat((class_token, x), dim=1)
        # print("in vit: ",x.shape)

        # Add th positional embedding to patch embedding with class token
        x = self.positional_embedding + x
        #print(x.shape)

        # dropout on patch + positional embedding
        x = self.embedding_dropout(x)

        # Pass embedding through Transformer Encoder stack
        x = self.transformer_encoder(x)

        # Pass 0th index of xthrough MLP head
        x = self.mlp_head(x[:,0])

        return x
