# logistic_regression

## About logistic_regression:

* logistic_regression is a machine learning project.

* logistic_regression is composed of two scripts, `logistic_reg.py` and `appli_logistic_reg.py`.

### About `logistic_reg.py`:

* `logistic_reg.py` trains thetas to predict class.

* It writes thetas in a file after trained them.

* `logistic_reg.py` use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to minimize cost function.

* I use [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) as cost function.

### About `appli_logistic_reg.py`:

* `appli_logistic_reg.py` use trained thetas to predict class acording to the new features passed in parameters.

* It writes new predictions in new file in `predictions`.

### About data.csv you can use.

* `data.csv` has two features and Y.

* You can create your own data.csv but the format must be [x1, x2, x3 ..., xm, Y]. x are the features, and Y is what you want predict and must be the last column in csv.

## What do you need to make logistic_regression work ?

* python >= 3.0

* [numpy](http://www.numpy.org/)

* [pandas](https://pandas.pydata.org/)

## Usage:

### `logistic_reg.py`

* `python3 logistic_reg.py [Data.csv]`.

### `appli_logistic_reg.py`

* `python3 appli_logistic_reg.py [Data.csv]`. Number of features must be the same as in data.csv for train.
