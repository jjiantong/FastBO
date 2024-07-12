# FastBO: Bayesian Optimization with Partial Evaluation for Hyperparameter Optimization

FastBO is implemented based on [Syne Tune](https://github.com/awslabs/syne-tune).

## How to use

Install FastBO: install everything in a virtual environment `st_venv`.Remember to activate 
this environment before working with FastBO. We also recommend building the
virtual environment from scratch now and then, in particular when you pull a new
release, as dependencies may have changed.

```bash
git clone https://github.com/jjiantong/FastBO.git
cd FastBO
python3 -m venv st_venv
. st_venv/bin/activate
pip install --upgrade pip
pip install -e '.[extra]'
```

Quick start for a simple example: run Bayesian Optimization with 4 workers on a local machine.
Please note that you have to report metrics from a training script
so that they can be communicated later to FastBO.
The training script for this example is ```examples/training_scripts/height_example/train_height_simple.py```.

```bash
cd examples
python3 lanuch_height_simple_bo.py
```

## Basic ideas





