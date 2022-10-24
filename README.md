
# Japanese Natural Language Inference Model
This repository provides the code for [Japanese NLI model](https://huggingface.co/cyberagent/xlm-roberta-large-jnli-jsick), a fine-tuned masked language model.

## Performance
The model showed performance comparable with those reported in [JGLUE](https://github.com/yahoojapan/JGLUE) [Kurihara et al. 2022] and [JSICK](https://github.com/verypluming/JSICK) [Yanaka and Mineshima 2022] papers, in terms of overall accuracy:

|              Model              | JGLUE-JNLI valid [%] | JSICK test [%] |
|:-------------------------------:|:----:|:-----:|
| [Kurihara et al. 2022]      | 91.9 |  N/A  |
| [Yanaka and Mineshima 2022] |  N/A |  89.1 |
| ours using both JNLI and JSICK  | 90.9 |  89.0 |

## References
- Hitomi Yanaka and Koji Mineshima. [Compositional Evaluation on Japanese Textual Entailment and Similarity](https://arxiv.org/abs/2208.04826). TACL2022.
- Kentaro Kurihara, Daisuke Kawahara, and Tomohide Shibata. [JGLUE: Japanese General Language Understanding Evaluation](https://aclanthology.org/2022.lrec-1.317/). LREC2022.
- Nils Reimers and Iryna Gurevych. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). EMNLP-IJCNLP2019.
- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/). ACL2020.

## Appendix: Hyperparameters

### random seeds
Yes, we tested only a single run :(
```python
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
```

### dataset order
1. JSICK
1. JGLUE

### labels
We converted string label into integer using the following mapping:
```python
label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
```

### CrossEncoder
We mimicked `batch_size=128` using gradient accumulation `32 * 4 = 128`.
```python
batch_size=32,
shuffle=True,
epochs=3,
accumulation_steps=4,
optimizer_params={'lr': 5e-5},
warmup_steps=math.ceil(0.1 * len(data)),
```
