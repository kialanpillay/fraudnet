# FraudNet

A feed-forward neural network architecture for credit card fraud detection.

---

## Installation
```
python3 -m pip install --user virtualenv
python3 -m venv venv
source ./venv/bin/activate
pip3 install -Ur requirements.txt
```
## Baseline
Run `main.py` with the `--baseline` flag to fit and evaluate the baseline classifiers.
```
python3 main.py --baseline
```

## Model Training

Run `main.py` with the following arguments to train the model with the desired hyperparameter configuration.
```
python3 main.py --train --num_epochs 100 --lr 0.001 --weight_decay 0.0001 --batch_size 32 --batch_norm --hidden_dim 32 --hidden layers 4
```

Run `main.py` with the `--train` flag to train the default (initial) feed-forward neural network model. 

Remove the `--verbose` flag to only print final validation set metrics. 

```
python3 main.py --train --verbose
```

Add the `--graph` flag to generate a train/validation set loss graph.

```
python3 main.py --train --graph
```

## Hyperparameter Tuning

Run `main.py` with the `--train` and `--hyper_opt` flags to perform tuning. Note the stochastic initialisation of the weights may lead to different results over different runs.  

```
python3 main.py --train --hyper_opt
```

### Optimal Configuration
| Hyperparameter      | Value   |
|---------------------|---------|
| Epochs              | 50      |
| Learning Rate       | 0.0005  |
| Weight Decay        | 0.00002 |
| Batch Size          | 64      |
| Batch Normalisation | True    |
| Hidden Nodes        | 64      |
| Hidden Layers       | 5       |

## Model Evaluation

Run `main.py` with the `--test` flag and to train and evaluate a model on the test set. Specify hyperparameters as above.

```
python3 main.py --test
```

Add the `--save` flag to write the model test set predictions to a `.csv` file.

```
python3 main.py --test --save
```


## Help

Run `main.py` with the `--help` flag to view the Help file.

```
python3 main.py --help
```