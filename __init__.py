import os

try:
    import numpy as np
    import colorsys
    from sklearn.cluster import KMeans
    import huggingface_hub
    import pandas as pd
    from PIL import Image
except Exception as e:
    print("###ComfyUI-PDiD-Nodes: Trying to install requirements...")
    os.system("pip install -U \"numpy<2\" \"scikit-learn\" \"huggingface_hub\" \"pandas\" \"Pillow\"")

from .nodes import CollectImageSize, GetImageMainColor, NearestSDXLResolutionby64, CheckAnimeCharacter, BlendTwoImages, ListOperation, RemoveSaturation


NODE_CLASS_MAPPINGS = {
    "Get image size": CollectImageSize,
    "Check Character Tag": CheckAnimeCharacter,
    "Nearest SDXL Resolution divided by 64": NearestSDXLResolutionby64,
    "Get Image Colors": GetImageMainColor,
    "Blend Images": BlendTwoImages,
    "List Operations": ListOperation,
    "Make Image Gray": RemoveSaturation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CollectImageSize": "Get Image Size",
    "CheckAnimeCharacter": "Check Character Tag",
    "NearestSDXLResolutionby64": "Nearest SDXL Resolution divided by 64",
    "GetImageMainColor": "Get Image Main Color",
    "BlendTwoImages": "Blend Images",
    "ListOperation": "List Operations",
    "RemoveSaturation": "Make Image Gray"
}

