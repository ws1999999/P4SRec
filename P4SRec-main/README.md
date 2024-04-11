# Introduction
Personalized Data Augmentation for Self-supervised Sequential Recommendation(P4SRec)

Source code for paper: https://github.com/ws1999999/P4SRec

# Implementation
## Requirements

Python = 3.8  
Pytorch >= 1.10.1  
tqdm == 4.65.0
gensim == 4.3.1
numpy == 1.24.4
## Datasets

Four prepared datasets are included in `data` folder.

## Train Model

To train P4SRec on `Sports` dataset, change to the `src` folder and run following command: 

You can train P4SRec on Beauty or Toys in a similar way.

The script will automatically train CoSeRec and save the best model found in validation set, and then evaluate on test set. You are expected to get following results after training:

```
'HIT@5': '0.0276', 'NDCG@5': '0.0189', 'HIT@10': '0.0422', 'NDCG@10': '0.0236', 'HIT@20': '0.0620', 'NDCG@20': '0.0285'
```


## Evaluate Model

You can directly evaluate a trained model on test set by running:

```
python main.py --data_name Sports --model_idx 0 --do_eval
```

We provide a model that trained on Sports_and_Games, Beauty, and Toys in `./src/output` folder. Please feel free to test is out.

# Acknowledgement
 - Transformer and training pipeline are implemented based on [CoSeRec](https://github.com/YChen1993/CoSeRec/tree/main/src). Thanks them for providing efficient implementation.

