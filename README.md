
# Connectivity-contrastive learning (CCL)

This code is the official implementation of

Hiroshi Morioka and Aapo Hyvärinen, Connectivity-Contrastive Learning: Combining Causal Discovery and Representation Learning for Multimodal Data. 
Proceedings of The 26th International Conference on Artificial Intelligence and Statistics (AISTATS2023).

If you are using pieces of the posted code, please cite the above paper. 


## Requirements

Python3

Pytorch


## Training

To train the models in the paper, run this command:

```train
python ccl_training.py
```

Set 'method' in the code either 'ccl' or 'cclalt'.

'ccl': Train by CCL

'cclalt': Train by CCLalt. Require 'pair' parameter

## Evaluation

To evaluate the trained model, run:

```eval
python ccl_evaluation.py
```
