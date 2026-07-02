# ATMs Review Paper Code

This repository contains code for running AMNet, LocalMapper, and SAMMNet on the random and EC-number benchmark splits.

## Setup

First, unzip all provided compressed files into the same parent directory.

Required packages include PyTorch, PyTorch Geometric, DGL/DGL-LifeSci, RDKit, pandas, NumPy, scikit-learn, tqdm, and matplotlib. GPU is recommended.

## Data

Prepared datasets are already included in each model folder:

```text
AMNet/dataset/
LocalMapper/data/
LocalMapper/datasets/
SAMMNet/SAMMNet-main/dataset/
```


## AMNet

Train both splits:

```bash
cd /work/zhang32/ATMs_review_paper_Code/AMNet
python train/train.py
```

Test both splits:

```bash
python test.py --dataset all
```

Main scripts:

```text
AMNet/train/train.py
AMNet/train/amnet_train.py
AMNet/test.py
```

Outputs:

```text
AMNet/saved_models/
AMNet/test_results/
```

## LocalMapper

Run from the scripts directory.

Train/test random split:

```bash
cd /work/zhang32/ATMs_review_paper_Code/LocalMapper/scripts
python auto_train_5_iterations.py -d custom_dataset -u user -g cuda:0 --num-iterations 5 --num-epochs 100
```

Train/test EC-number split:

```bash
python auto_train_5_iterations.py -d EC_split_N2 -u user -g cuda:0 --num-iterations 5 --num-epochs 100
```

Evaluate accuracy:

```bash
python calculate_accuracy.py --dataset all --model-iteration 5 --user-name user --device cuda:0
```

Main scripts:

```text
LocalMapper/scripts/auto_train_5_iterations.py
LocalMapper/scripts/Train.py
LocalMapper/scripts/Test.py
LocalMapper/scripts/calculate_accuracy.py
```

Outputs:

```text
LocalMapper/models/
LocalMapper/outputs/
```

## SAMMNet

Run from the training directory.

Train random split:

```bash
cd /work/zhang32/ATMs_review_paper_Code/SAMMNet/SAMMNet-main/training
python multitask_train.py
```

Train EC-number split:

```bash
torchrun --nproc_per_node=1 multitask_train_EC_split_N2.py
```

Evaluate both splits:

```bash
python evaluate_accuracy.py --datasets random EC_split_N2 --top-k 1 3 5 10 --device cuda
```

Main scripts:

```text
SAMMNet/SAMMNet-main/training/multitask_train.py
SAMMNet/SAMMNet-main/training/multitask_train_EC_split_N2.py
SAMMNet/SAMMNet-main/training/evaluate_accuracy.py
```

Outputs:

```text
SAMMNet/SAMMNet-main/training/multitask_results/
SAMMNet/SAMMNet-main/training/multitask_topk_metrics/
```

## Plots

Accuracy bars:

```bash
cd /work/zhang32/ATMs_review_paper_Code/Graph/accuracy
python plot_accuracy_bar.py
```

Similarity t-SNE:

```bash
cd /work/zhang32/ATMs_review_paper_Code/Graph/Similarity
python plot_similarity.py
```
