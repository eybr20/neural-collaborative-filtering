# Neural Collaborative Filtering for movie rating prediction

Predicting missing movie ratings (0.5–5 stars) from a sparse user × movie matrix. The project compares classic matrix factorization with the **Neural Collaborative Filtering** family (GMF, MLP, NeuMF — [He et al., 2017](https://arxiv.org/abs/1708.05031)), adapts them to explicit ratings, and tunes NeuMF with a 960-configuration grid search.

*Data Science Lab project, M2 IASD (PSL University).*

## Results

Test RMSE on the held-out ratings (lower is better). Final models averaged over 3 seeds.

| Model | Validation RMSE | Test RMSE | Parameters |
|---|---|---|---|
| Matrix factorization (baseline) | 0.9165 ± 0.0017 | 0.9054 ± 0.0006 | 50,311 |
| NeuMF as in the paper | 0.8968 ± 0.0014 | 0.8832 ± 0.0010 | 226,379 |
| **NeuMF, tuned** | **0.8848 ± 0.0008** | **0.8741 ± 0.0007** | **50,481** |

The tuned NeuMF beats the paper's configuration with **4.5× fewer parameters**.

## Approach

1. **Adapting NCF to explicit ratings.** Identity output instead of a sigmoid, MSE loss, output bias initialized at the mean rating.
2. **Model comparison.** MF, GMF, MLP and NeuMF (from scratch and with pre-trained branches), all trained the same way (mini-batches, Adam) so differences reflect the model only.
3. **Regularization, one technique at a time.** User and item biases, weight decay (AdamW), dropout in the MLP tower, smaller models and shared embeddings between the GMF and MLP branches.
4. **Grid search** over NeuMF (`ncf_search.py`): embedding size, MLP architecture, dropout, weight decay, embedding sharing, learning rate and batch size. Top configurations re-run with several seeds to check that the ranking is stable.

## Key findings

- **User and item biases matter most for MF.** Validation RMSE drops from 0.979 to 0.916: much of what is predictable is that some users rate everything high and some movies are liked by everyone.
- **Shared embeddings win.** All top-10 configurations share user and item vectors between the two branches. With very few ratings per movie, every extra vector is mostly noise to learn.
- **Bigger is not better.** Deeper or wider MLP towers rank last, and small models match large ones with 2–5× fewer parameters.
- **Differences are compared to noise.** Gaps below the ~0.008 validation noise are reported as ties rather than wins.

## Repository structure

```
├── baseline.ipynb          # Matrix factorization baseline
├── ncf.ipynb               # GMF, MLP, NeuMF: experiments, regularization study, final evaluation
├── ncf_search.py           # Grid search / multi-seed evaluation (CLI, Slurm-ready)
├── generate.py             # Trains the final model and fills the ratings table
├── results/search1/        # Grid search settings and results (960 configurations)
└── ratings_per_movie.png   # Data exploration
```

## Usage

```bash
pip install -r requirements.txt

# Data: place ratings_train.npy, ratings_test.npy and namesngenre.npy in datasets/
# (course data, not included in this repository)

# Best configuration: validation, then retrain and score on the test set
python ncf_search.py --model NeuMF --factors 8 --layers 16 8 --dropout 0.5 \
    --weight-decay 0 --share --lr 0.003 --test

# Grid search
python ncf_search.py --search --out results/search1

# Fill a ratings table with the final model
python generate.py --name datasets/ratings_train.npy
```

## Tech stack

Python · PyTorch · NumPy · pandas · Matplotlib
