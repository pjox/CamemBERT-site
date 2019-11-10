+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 110  # Order that this section will appear.

title = "Download"
subtitle = ""

[design]
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns = "1"

[design.background]
  # Apply a background color, gradient, or image.
  #   Uncomment (by removing `#`) an option to apply it.
  #   Choose a light or dark text color by setting `text_color_light`.
  #   Any HTML color name or Hex value is valid.

  # Background color.
  # color = "navy"
  
  # Background gradient.
  # gradient_start = "DeepSkyBlue"
  # gradient_end = "SkyBlue"
  
  # Background image.
  image = ""  # Name of image in `static/img/`.
  image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.

  # Text color (true=light or false=dark).
  text_color_light = false

[design.spacing]
  # Customize the section spacing. Order is top, right, bottom, left.
  padding = ["20px", "0", "20px", "0"]

[advanced]
 # Custom CSS. 
 css_style = ""
 
 # CSS class.
 css_class = ""
+++

## Pre-trained models

| Model       | #params | vocab size | Download                                                                                 |
| ----------- | ------- | ---------- | ---------------------------------------------------------------------------------------- |
| `CamemBERT` | 110M    | 32k        | [camembert.v0.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert.v0.tar.gz) |


## Example usage

##### Load CamemBERT from torch.hub (PyTorch >= 1.1):
```python
import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert.v0')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load CamemBERT (for PyTorch 1.0 or custom models):
```python
# Download camembert model
wget https://dl.fbaipublicfiles.com/fairseq/models/camembert.v0.tar.gz
tar -xzvf camembert.v0.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import CamembertModel
camembert = CamembertModel.from_pretrained('/path/to/camembert.v0')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Filling masks:
```python
masked_line = 'Le camembert est <mask> :)'
camembert.fill_mask(masked_line, topk=5)
# [('Le camembert est délicieux :)', 0.4909118115901947, ' délicieux'),
#  ('Le camembert est excellent :)', 0.10556942224502563, ' excellent'),
#  ('Le camembert est succulent :)', 0.03453322499990463, ' succulent')]
```

##### Extract features from Camembert:
```python
# Extract the last layer's features
line = "J'aime le camembert!"
tokens = camembert.encode(line)
last_layer_features = camembert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 10, 768])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```
