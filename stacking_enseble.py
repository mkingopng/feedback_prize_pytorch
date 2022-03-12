"""
https://towardsdatascience.com/ensembles-the-almost-free-lunch-in-machine-learning-91af7ebe5090
"""
import numpy as np
import pandas as pd


def accuracy(predictions, targets, one_hot_targets=True):
    """Compute accuracy given arrays of predictions and targets.
    Parameters
    predictions: np.array
        (num examples, num_classes) of predicted class probabilities/scores
    targets: np.array
        (num examples, num_classes) of one hot encoded true class labels if
        'one_hot_targets' is True, or true class indices if it is False
    one_hot_targets: bool
        whether the target are in one-hot or class index format. Default is
        True
    Returns
    -------
    accuracy: float
        accuracy of predictions
    """
    if one_hot_targets:
        return (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()
    else:
        return (predictions.argmax(axis=1) == targets).mean()


def cross_entropy(predictions, targets, epsilon=1e-8):
    """Compute cross entropy given predictions as class probabilities and one-
    hot encoded ground truth labels.
    Parameters
    ----------
    predictions: np.array
        (num examples, num_classes) of predicted class probabilities
    targets: np.array
        (num examples, num_classes) of one-hot encoded true class labels
    epsilon: float
     a constant to clip predicted probabilities to avoid taking log of zero
    Returns
    -------
    cross_entropy: float
        cross entropy of the inputs
    """
    predictions = np.clip(predictions, epsilon, 1-epsilon)
    return (-np.log(predictions) * targets).sum(axis=1).mean()


def ensemble_selector(loss_function, y_hats, y_true, init_size=1,
                      replacement=True, max_iter=100):
    """Implementation of the algorithm of Caruana et al. (2004) 'Ensemble
    Selection from Libraries of Models'. Given a loss function mapping
    predicted and ground truth values to a scalar along with a dictionary of
    models with predicted and ground truth values, constructs an optimal
    ensemble minimizing ensemble loss, by default allowing models to appear
    several times in the ensemble.
    Parameters
    ----------
    loss_function: function
        accepting two arguments - numpy arrays of predictions and true values -
        and returning a scalar
    y_hats: dict
        with keys being model names and values being numpy arrays of predicted
        values
    y_true: np.array
        numpy array of true values, same for each model
    init_size: int
        number of models in the initial ensemble, picked by the best loss.
        Default is 1
    replacement: bool
        whether the models should be returned back to the pool of models once
        added to the ensemble. Default is True
    max_iter: int
        number of iterations for selection with replacement to perform. Only
        relevant if 'replacement' is True, otherwise iterations continue until
        the dataset is exhausted i.e.
        min(len(y_hats.keys())-init_size, max_iter). Default is 100
    Returns
    -------
    ensemble_loss: pd.Series
        with loss of the ensemble over iterations
    model_weights: pd.DataFrame
        with model names across columns and ensemble selection iterations
        across rows. Each value is the weight of a model in the ensemble
    """
    # Step 1: compute losses
    losses = dict()
    for model, y_hat in y_hats.items():
        losses[model] = loss_function(y_hat, y_true)

    # Get the initial ensemble comprising the best models
    losses = pd.Series(losses).sort_values()
    init_ensemble = losses.iloc[:init_size].index.tolist()

    # Compute its loss
    if init_size == 1:
        # Take the best loss
        init_loss = losses.loc[init_ensemble].values[0]
        y_hat_avg = y_hats[init_ensemble[0]].copy()
    else:
        # Average the predictions over several models
        y_hat_avg = np.array(
            [y_hats[mod] for mod in init_ensemble]).mean(axis=0)
        init_loss = loss_function(y_hat_avg, y_true)

    # Define the set of available models
    if replacement:
        available_models = list(y_hats.keys())
    else:
        available_models = losses.index.difference(init_ensemble).tolist()
        # Redefine maximum number of iterations
        max_iter = min(len(available_models), max_iter)

    # Sift through the available models keeping track of the ensemble loss
    # Redefine variables for the clarity of exposition
    current_loss = init_loss
    current_size = init_size

    loss_progress = [current_loss]
    ensemble_members = [init_ensemble]
    for i in range(max_iter):
        # Compute weights for predictions
        w_current = current_size / (current_size + 1)
        w_new = 1 / (current_size + 1)

        # Try all models one by one
        tmp_losses = dict()
        tmp_y_avg = dict()
        for mod in available_models:
            tmp_y_avg[mod] = w_current * y_hat_avg + w_new * y_hats[mod]
            tmp_losses[mod] = loss_function(tmp_y_avg[mod], y_true)

        # Locate the best trial
        best_model = pd.Series(tmp_losses).sort_values().index[0]

        # Update the loop variables and record progress
        current_loss = tmp_losses[best_model]
        loss_progress.append(current_loss)
        y_hat_avg = tmp_y_avg[best_model]
        current_size += 1
        ensemble_members.append(ensemble_members[-1] + [best_model])

        if not replacement:
            available_models.remove(best_model)

    # Organize the output
    ensemble_loss = pd.Series(loss_progress, name="loss")
    model_weights = pd.DataFrame(index=ensemble_loss.index,
                                 columns=y_hats.keys())
    for ix, row in model_weights.iterrows():
        weights = pd.Series(ensemble_members[ix]).value_counts()
        weights = weights / weights.sum()
        model_weights.loc[ix, weights.index] = weights

    return ensemble_loss, model_weights.fillna(0).astype(float)


np.random.seed(1)

# Predictions are mean-zero uncorrelated normals...
y_hats = np.random.standard_normal((1000, 10))

# ... with variance decreasing linearly from model to model
st_devs = np.array(
    [k ** 0.5 for k in range(1, y_hats.shape[1] + 1, 1)]
    )[::-1]
y_hats *= st_devs

# Targets are zeros
y_true = np.zeros((y_hats.shape[0],))


# Optimization objective is to minimize MSE
def mse_loss(predictions, targets):
    return ((predictions-targets)**2).mean()


# The ensemble building function accepts a dictinary with predictions
y_hats_dict = {"M" + str(i): y_hats[:, i] for i in range(y_hats.shape[1])}

# Find the optimal ensemble
ensemble_loss, model_weights = ensemble_selector(
    loss_function=mse_loss, y_hats=y_hats_dict, y_true=y_true,
    init_size=1, replacement=True, max_iter=1000)

# Get the weights corresponding to the lowest ensemble loss
estimated = model_weights.loc[
    ensemble_loss[ensemble_loss == ensemble_loss.min()].index[0]]

# Compute theoretically optimal weights (assuming the variances are known)
variances = st_devs ** 2
theoretical = pd.Series((1 / variances) / (1 / variances).sum(), index=model_weights.columns)

# Construct a naive ensemble by simply averaging the predictions
simple_ensemble_loss = mse_loss(y_hats.mean(axis=1), y_true)

ensemble_loss, model_weights = ensemble_selector(
  loss_function=cross_entropy, y_hats=y_hats_val,
  y_true=y_true_one_hot_val, init_size=1, replacement=True, max_iter=10
  )

ensemble_acc, model_weights_acc = ensemble_selector(
    loss_function=lambda p, t: -accuracy(p, t),  # - for minimization
    y_hats=y_hats_val, y_true=y_true_one_hot_val,
    init_size=1, replacement=True, max_iter=10
    )
ensemble_acc = -ensemble_acc  # back to positive domain