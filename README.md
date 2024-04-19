# 0. Overview
<p align="center">
  <img src="https://github.com/cianeastwood/egfr/blob/main/assets/roc_curve.png?raw=true" width="400" alt="ROC Curve" />
</p>

## 0.1 Context
During a drug discovery program, one essential task is screening a library of compounds to find molecules that can bind to and potentially inhibit the activity of a target protein (referred to as "potency"). To reduce costs and prioritize molecules
efficiently, virtual in-silico screening is used as an initial step. Ligand-based machine learning models, utilizing molecular structures as inputs, are common methods for virtual screening. In this task, a dataset of 4.6k compounds tested against the _Epidermal Growth Factor Receptor_ (EGFR) kinase, a target associated with various cancers, is provided. **The goal is to build a machine learning model and accompanying inference pipeline to predict the potency value (pIC50) for novel compounds targeting EGFR.**

## 0.2 Tasks
- Your predictor should be able to detect active compounds that have a value of
pIC50>8.
- You should report appropriate performance metrics.
- You should outline the decisions and motivations behind your chosen approach.

## 0.3 Data
- The dataset is available in the `data` folder and at [this link](https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/data/EGFR_compounds_lipinski.csv).

## 0.4 Starter code / tutorial
- [This tutorial](https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html) shows how to train ML predictor on the above dataset. We use some of these code snippets in this repo.


# 1. Installation
We used Python 3.12. The molfeat package can take a long time to install via conda or pip, so we recommend first using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install some base packages (where needed), and then using pip to install the rest.
```bash
mamba install -c conda-forge numpy pandas scikit-learn matplotlib datamol molfeat
```

and then

```bash
pip install tqdm scikit-learn tokenizers transformers fcd-torch xgboost tabulate
```

**Alternatively**, you can install all the packages at once using pip (though this may take longer):
```bash
pip install -r requirements.txt
```

# 2. Usage

## 2.1 Single runs
### 2.1.1 Training a model
```bash
python train.py --model_name linear --fingerprint fcfp --pretr_feat chemberta --sweep_name test_run --verbose
```

### 2.1.2 Evaluating a model (on the test set)
```bash
python evaluate.py --results_dir results/test_run
```

## 2.2 Multiple runs/sweeps (to reproduce the results)
### 2.2.1 Create sweep commands
Create a text file of commands `job_scripts/my_sweep.txt` with the following command (_note:_ if running these on a cluster, you'll need to specify the _absolute_ paths to your data and output directories, otherwise relative paths are fine):
```bash
python -m job_scripts.my_sweep
```

### 2.2.2 Run the commands
Run the commands in the text file using `jobs_scripts/run_local.py`. To do so on a **local machine** (warning: may take 30+ minutes), use:
```bash
python -m job_scripts.run_local --commands_fpath job_scripts/jobs/my_sweep.txt
```

**OR**, to do so via a **slurm cluster**, `job_scripts/submit_jobs.py` provides a useful starting point (edit with the details of your cluster). After installing [submitit](https://github.com/facebookincubator/submitit) via pip, the following command will then run the commands in the text file by submitting them to a slurm cluster:
```sh
python -m job_scripts.run_cluster -c job_scripts/jobs/my_sweep.txt
```

### 2.2.3 View results
Validation-set results will have been saved to `results/my_sweep` (unless the `--results_dir` or `--sweep_name` flags were changed in step 2.2.1). To collect these results, select the best sweep for each model (based on validation-set performance), and then report the test-set performance, run the following command (optionally plotting results with the `--plot` flag):
```bash
python evaluate.py --results_dir results/my_sweep --plot
```

### 2.2.4 Further analysing/filter results
The current setup ensures that **the test set is not used for model selection or hyperparameter tuning, but only for final evaluation**. However, it is possible to get some insights into test-set performance by filtering the results using the `--arg_values` flag of `evaluate.py`. For example, to view the best run for each model **when using the pretrained ChemBERTa features**, you can run:
```bash
python evaluate.py --results_dir results/my_sweep --arg_values pretr_feat=chemberta
```

Or, to additionally filter results to those **also using fcfp fingerprints**, you can run:
```bash
python evaluate.py --results_dir results/my_sweep --arg_values pretr_feat=chemberta,fp_methods=fcfp
```


# 3. Design choices
The main factors affecting the design choices were:
- **Dataset size**: The dataset contains 4.6k compounds, which is relatively small. This means that models with many parameters (e.g., neural networks) may overfit and that (K-fold) cross-validation is essential to get a good estimate of the model's performance. For this reason, we focused on smaller sklearn models, which are easier to cross-validate, and which have fewer parameters to tune.
- **Imbalanced classes**: The dataset is severely imbalanced, with only 10% of compounds being active (pIC50>8). This means that:
    - _Performace metric:_ Accuracy is not a good metric, and metrics like balanced accuracy (per class), F2-score, precision, recall, and the area under the ROC curve (AUC) are more informative.
    - _Loss function:_ Standard cross entropy may not be the best loss function, as it may be biased towards the majority class. For this reason, we used "balanced" losses for all models (these re-weight sample losses based on the class imbalance or positive-negative ratio). See the `get_model_and_hparams()` function in `train.py` for more details.
    - _Data splitting:_ The dataset should be stratified when splitting into training and test sets (see our use of `train_test_split()` in `data/loader.py`), and when performing cross-validation (see the `kfold_cross_validate()` function of `train.py`). 
- **Screening task**: The goal is to screen a library of compounds to find those that are active against the EGFR kinase. We understood this to mean that false negatives were more costly than false positives, i.e., that **recall is more important than precision**, since we want to avoid missing active compounds for subsequent stages of the pipeline. This informed our choice of performance metric, the F2-score, which weights recall twice as much as precision.


## 3.1 Data preprocessing
The data is preprocessed in the `data/loader.py` file. The main choices were:
- **Features to use**: We chose to use fingerprints and pretrained features, both extracted from the SMILES string, and discard additional dataset features like molecular weight, number of hydrogen bond donors/acceptors, etc. This was mainly because we were not familiar with these features, and some initial tests indicated that they were not that helpful. However, with more thought and preprocessing, these features may be helpful (section 5.1 below).
- **Consistent train-test split**: We save the indices of the train-test split to ensure that the same split is used for all models and runs. This is important to ensure that the models are comparable, and that the test set is not used for model selection or hyperparameter tuning.
- **Save/load features**: We save both fingerprint and pretrained features to disk after preprocessing, to avoid recomputing them for each model. This is particularly important for the pretrained features, which can take a long time to compute (e.g., ChemBERTa).


## 3.2 Models
As explained above, we chose to use smaller sklearn models, which are easier to cross-validate and have fewer parameters to tune. In particular, we used the following models:
- **Linear**: A simple linear model, which is easy to interpret and has few parameters to tune. We used the `LogisticRegression` class from sklearn.
- **Random forest**: A tree-based ensemble model that can capture non-linear relationships in the data, and reduce variance by averaging across models in the ensemble (we fixed `n_estimators=100`). We used the `RandomForestClassifier` class from sklearn.
- **Gradient-boosted trees**: Another tree-based ensemble model that can capture non-linear relationships and reduce variance. We used the `XGBClassifier` class from the xgboost library, and also fixed `n_estimators=100`.


## 3.3 Performance metrics
As explained above, we chose a performance metric which: **a. works well for highly imbalanced classes; and b. weights recall more than precision**. We chose the **F2-score**, where recall is weighted twice as much as precision. While this metric is used for model selection, we also report other metrics on the test set, including accuracy, balanced accuracy, precision, recall, average precision, specificity, AUC, and F1-score, to give a more complete picture of the model's performance. We also plot the ROC curve (see above) to visualize the trade-off between true positive rate and false positive rate.


## 3.4 Hyperparameter tuning
We performed two different types of hyperparameter tuning or sweeps:
- **Model hyperparameters**: For the linear model, we tuned the regularization strength `C` (inverse of weight decay), and for the random forest and gradient-boosted trees, we tuned the maximum depth of the trees `max_depth`. These were selected using a grid search, with the best model being selected based on the F2-score with k-fold cross-validation on the training set. See the `get_model_and_hparams()` function in `train.py` for more details.
- **Features to use**: We also performed a sweep over the features to use, including different fingerprints (fcfp, ecfp, maccs) and pretrained features (FCD, ChemBERTa). These were also selected using a grid search, with the best model being selected based on the F2-score with k-fold cross-validation on the training set. See `job_scripts/my_sweep.py` for more details.


# 4. Results
The following table shows the test-set performance for the best-performing models of each type (linear, random forest, gradient-boosted trees). For each model type, the best model is selected based on the F2-score with k-fold cross-validation on the training set, and then the test-set performance is reported for all metrics. Here we see that gradient-boosted trees (GBT) performed best.

**Test-set results:**

| model   | fingerpr   | pretr_feat   |   acc |   bal_acc |   prec |   recall |   avg_prec |   spec |   auc |   f1 |   f2 |
|---------|------------|--------------|-------|-----------|--------|----------|------------|--------|-------|------|------|
| gbt     | fcfp       |              |  0.80 |      0.83 |   0.50 |     0.88 |       0.46 |   0.79 |  0.91 | 0.64 | 0.76 |
| linear  | fcfp       |              |  0.83 |      0.83 |   0.53 |     0.83 |       0.48 |   0.82 |  0.91 | 0.65 | 0.75 |
| rf      | maccs,fcfp |              |  0.77 |      0.81 |   0.46 |     0.87 |       0.42 |   0.75 |  0.90 | 0.60 | 0.74 |

Surprisingly, the best models do not use pretrained molecular features (e.g., ChemBERTa, FCD). This is perhaps due to the small size of the dataset, making it difficult to robustly use these features. To investigate, we filtered the results to show only those using ChemBERTa pretrained features, finding that the linear model with ChemBERTa pretrained features actually performs best on the test set (F2-score of 0.77), but was not selected during the k-fold cross-validation due to a worse score there. To illustrate this point, the tables below show the test-set performance for the best-performing models of each type when: a. constrained to use ChemBERTa features; and b. _not_ constrained (as above). Both tables show the F2-scores on both the test data (`f2`) and the validation data (`f2_val`), demonstrating why the linear model with ChemBERTa features was not selected as the best model above (and the danger of fitting to the test set, i.e., selecting models based on test-set performance). This suggests that with more data to train and validate models, a linear model with ChemBERTa pretrained features may be best choice.

**Constrained to use ChemBERTa features:**
| model   | fingerpr   | pretr_feat   |   acc |   bal_acc |   prec |   recall |   avg_prec |   spec |   auc |   f1 |   f2 |   f2_val |
|---------|------------|--------------|-------|-----------|--------|----------|------------|--------|-------|------|------|----------|
| linear  | fcfp       | chemberta    |  0.83 |      0.84 |   0.55 |     0.85 |       0.49 |   0.83 |  0.91 | 0.67 | 0.77 |     0.70 |
| gbt     | ecfp       | chemberta    |  0.77 |      0.80 |   0.46 |     0.84 |       0.41 |   0.76 |  0.87 | 0.59 | 0.72 |     0.65 |
| rf      | ecfp       | chemberta    |  0.70 |      0.77 |   0.39 |     0.88 |       0.36 |   0.66 |  0.85 | 0.54 | 0.70 |     0.67 |

**Not constrained:**
| model   | fingerpr   | pretr_feat   |   acc |   bal_acc |   prec |   recall |   avg_prec |   spec |   auc |   f1 |   f2 |   f2_val |
|---------|------------|--------------|-------|-----------|--------|----------|------------|--------|-------|------|------|----------|
| gbt     | fcfp       |              |  0.80 |      0.83 |   0.50 |     0.88 |       0.46 |   0.79 |  0.91 | 0.64 | 0.76 |     0.71 |
| linear  | fcfp       |              |  0.83 |      0.83 |   0.53 |     0.83 |       0.48 |   0.82 |  0.91 | 0.65 | 0.75 |     0.71 |
| rf      | maccs,fcfp |              |  0.77 |      0.81 |   0.46 |     0.87 |       0.42 |   0.75 |  0.90 | 0.60 | 0.74 |     0.69 |


# 5. Possible improvements

## 5.1 Data
- **Use more dataset features**: The current models have access to fingerprints (fcfp, ecfp, maccs) and pretrained features (FCD, ChemBERTa). One could also make use of some of the original data features, such as molecular weight, number of hydrogen bond donors/acceptors, etc., if they are predictive of the target variable. These may also need to be preprocessed, e.g., by standardizing continuous features, or converting categorical features to one-hot encodings.
- **Use different pretrained models**: One could also try other pretrained models like Microsoft's Graphormer. This is provided by the molfeat package, but requires an older version of python so was ignored in this version.


## 5.2 Models
- **Use more models**: The current models are linear regression, random forest, and gradient-boosted trees. One could also try other models like neural networks. However, these may require more data to train, or strong regularization to prevent overfitting. For neural networks, one could start with a simple _multi-layer perceptron_ (MLP). The easiest way to add this would be to add a new model class `MLP()` which implements the sklearn methods like `fit()`, `predict()`, etc., and trains these models using the PyTorch library.
- **Fine-tune pretrained models**: The current models use the pretrained features as is, but one could also fine-tune the pretrained models on the current dataset to see if this improves performance. However, this may require collecting more data as the current dataset is relatively small, containing only 4.6k compounds (~3.7k for training and validation).

## 5.3 Training
- **Multiple random seeds**: The current models are trained with a single random seed, but one could also train models with multiple random seeds to get a better estimate of the model's performance. This could be done by training models multiple times with different seeds, and providing the mean and standard deviation of the performance metric(s).
