+++
# A Demo section created with the Blank widget.
# Any elements can be added in the body: https://sourcethemes.com/academic/docs/writing-markdown-latex/
# Add more sections by duplicating this file and customizing to your requirements.

widget = "blank"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 100  # Order that this section will appear.

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



# Pre-trained models

CamemBERT is available in [github.com/huggingface/transformers](https://github.com/huggingface/transformers/) and [https://github.com/pytorch/fairseq/](https://github.com/pytorch/fairseq/tree/master/examples/camembert)

## fairseq

<div style="margin: 0 auto; width: 994px;">

<!-- For some reasons table were not rendered correctly, this hack (along with the layouts/shortcodes snippet) solves it -->
{{<table "table table-striped table-bordered">}}
| Model                          | #params | Download                                                                                                                 | Arch. | Training data                                                 |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------ | ----- | ------------------------------------------------------------- |
| `camembert` / `camembert-base` | 110M    | [camembert-base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base.tar.gz)                             | Base  | [OSCAR](https://oscar-corpus.com) (138 GB of text)            |
| `camembert-large`              | 335M    | [camembert-large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-large.tar.gz)                           | Large | CCNet (135 GB of text)                                        |
| `camembert-base-ccnet`         | 110M    | [camembert-base-ccnet.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-ccnet.tar.gz)                 | Base  | CCNet (135 GB of text)                                        |
| `camembert-base-wikipedia-4gb` | 110M    | [camembert-base-wikipedia-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-wikipedia-4gb.tar.gz) | Base  | Wikipedia (4 GB of text)                                      |
| `camembert-base-oscar-4gb`     | 110M    | [camembert-base-oscar-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-oscar-4gb.tar.gz)         | Base  | Subsample of [OSCAR](https://oscar-corpus.com) (4 GB of text) |
| `camembert-base-ccnet-4gb`     | 110M    | [camembert-base-ccnet-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-ccnet-4gb.tar.gz)         | Base  | Subsample of CCNet (4 GB of text)                             |

{{</table>}}
</div>

##### Load CamemBERT from torch.hub (PyTorch >= 1.1):
```python
import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load CamemBERT (for PyTorch 1.0 or custom models):
Download camembert model
```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/camembert-base.tar.gz
tar -xzvf camembert-base.tar.gz
```
Load the model in fairseq
```python
from fairseq.models.roberta import CamembertModel
camembert = CamembertModel.from_pretrained('./camembert-base/')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Filling masks:
```python
masked_line = 'Le camembert est <mask> :)'
camembert.fill_mask(masked_line, topk=3)
# [('Le camembert est délicieux :)', 0.4909118115901947, ' délicieux'),
#  ('Le camembert est excellent :)', 0.10556942224502563, ' excellent'),
#  ('Le camembert est succulent :)', 0.03453322499990463, ' succulent')]
```

##### Extract features from Camembert:
```python
# Extract the last layer's features
line = "J'aime le camembert !"
tokens = camembert.encode(line)
last_layer_features = camembert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 10, 768])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```

## Transformers

```python
import torch

from transformers.modeling_camembert import CamembertForMaskedLM
from transformers.tokenization_camembert import CamembertTokenizer


def fill_mask(masked_input, model, tokenizer, topk=5):
    # Adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
    assert masked_input.count("<mask>") == 1
    input_ids = torch.tensor(tokenizer.encode(masked_input, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    logits = model(input_ids)[0]  # The last hidden-state is the first element of the output tuple
    masked_index = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
    logits = logits[0, masked_index, :]
    prob = logits.softmax(dim=0)
    values, indices = prob.topk(k=topk, dim=0)
    topk_predicted_token_bpe = " ".join(
        [tokenizer.convert_ids_to_tokens(indices[i].item()) for i in range(len(indices))]
    )
    masked_token = tokenizer.mask_token
    topk_filled_outputs = []
    for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(" ")):
        predicted_token = predicted_token_bpe.replace("\u2581", " ")
        if " {0}".format(masked_token) in masked_input:
            topk_filled_outputs.append(
                (
                    masked_input.replace(" {0}".format(masked_token), predicted_token),
                    values[index].item(),
                    predicted_token,
                )
            )
        else:
            topk_filled_outputs.append(
                (masked_input.replace(masked_token, predicted_token), values[index].item(), predicted_token,)
            )
    return topk_filled_outputs


tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
model.eval()

masked_input = "Le camembert est <mask> :)"
print(fill_mask(masked_input, model, tokenizer, topk=3))
```

## Citation
If you use our work, please cite:

```bibtex
@inproceedings{martin2020camembert,
  title={CamemBERT: a Tasty French Language Model},
  author={Martin, Louis and Muller, Benjamin and Ortiz Su{\'a}rez, Pedro Javier and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^\i}t},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

