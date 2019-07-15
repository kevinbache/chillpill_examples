"""This is the main training module which is run by the Cloud AI Platform jobs that are launched from
`run_cloud_tuning_job.py`"""
from sklearn import datasets
from tensorflow import keras

from chillpill import params, callbacks
from chillpill_examples.hp_tuning_from_train_fn import models


class MyParams(params.ParameterSet):
    """Define parameters names and default values for development and typechecked autocomplete."""
    # data parameters
    num_samples = 1000
    num_classes = 2
    valid_portion = 0.15
    random_state = 42
    # model parameters
    activation = 'relu'
    num_layers = 2
    num_neurons = 16
    dropout_rate = 0.5
    # training parameters
    learning_rate = 0.01
    batch_size = 16
    num_epochs = 10
    metrics = ['accuracy']


def train_fn(hp: MyParams):
    # generate data
    x, y = datasets.make_classification(
        n_samples=hp.num_samples,
        random_state=hp.random_state,
        n_classes=hp.num_classes,
    )
    y = keras.utils.to_categorical(y, hp.num_classes)

    # generate model
    model = models.build_and_compile_model(hp, x)

    # train model
    history = model.fit(
        x, y,
        batch_size=hp.batch_size,
        validation_split=hp.valid_portion,
        epochs=hp.num_epochs,
        verbose=2,
        callbacks=[callbacks.GoogleCloudAiCallback()]
    )

    return history
