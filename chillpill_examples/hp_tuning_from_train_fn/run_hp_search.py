"""This module runs a distributed hyperparameter tuning job on Google Cloud AI Platform."""
from pathlib import Path
import numpy as np

import chillpill
from chillpill import params, search

from chillpill_examples.hp_tuning_from_train_fn import train


if __name__ == '__main__':
    # Create a Cloud AI Platform Hyperparameter Search object
    search = search.HyperparamSearchSpec(
        max_trials=10,
        max_parallel_trials=5,
        max_failed_trials=2,
        hyperparameter_metric_tag='val_acc',
    )

    # Add parameter search ranges for this problem.
    my_param_ranges = train.MyParams(
        activation=params.Categorical(['relu', 'tanh']),
        num_layers=params.Integer(min_value=1, max_value=3),
        num_neurons=params.Discrete(np.logspace(2, 8, num=7, base=2)),
        dropout_rate=params.Double(min_value=-0.1, max_value=0.9),
        learning_rate=params.Discrete(np.logspace(-6, 2, 17, base=10)),
        batch_size=params.Integer(min_value=1, max_value=128),
    )
    search.add_parameters(my_param_ranges)

    # Run hyperparameter search job
    search.run_from_trian_fn(
        train_fn=train.train_fn,
        train_params_type=train.MyParams,
        additional_package_root_dirs=[Path(chillpill.__file__).parent.parent],
        cloud_staging_bucket='chillpill-staging-bucket',
        gcloud_project_name='kb-experiment',
        region='us-central1',
    )

