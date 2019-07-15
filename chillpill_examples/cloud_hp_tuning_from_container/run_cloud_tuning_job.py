"""This module runs a distributed hyperparameter tuning job on Google Cloud AI Platform."""
import os
from pathlib import Path
import subprocess

import numpy as np

from chillpill import params
from chillpill import search
from chillpill_examples.cloud_hp_tuning_from_container import train

if __name__ == '__main__':
    GCLOUD_PROJECT_NAME = 'kb-experiment'
    CONTAINER_IMAGE_URI = f'gcr.io/{GCLOUD_PROJECT_NAME}/chillpill:cloud_hp_tuning_example'
    GCLOUD_BUCKET_NAME = 'kb-bucket'

    env = os.environ.copy()
    env['PROJECT_ID'] = GCLOUD_PROJECT_NAME
    env['IMAGE_URI'] = CONTAINER_IMAGE_URI
    env['BUCKET_NAME'] = GCLOUD_BUCKET_NAME

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

    # Call a bash script to build a docker image for this repo, submit it to the docker registry defined in the script
    # and run a training job on the Cloud AI Platform using this container and these hyperparameter ranges.
    this_dir = Path(__file__).resolve().parent
    retcode = subprocess.call([this_dir / 'build_push.sh'], env=env)
    if retcode:
        raise ValueError(f"Got returncode: {retcode}")

    search.run_from_container(
        gcloud_project_name=GCLOUD_PROJECT_NAME,
        container_image_uri=CONTAINER_IMAGE_URI,
        static_args={'bucket_id': GCLOUD_BUCKET_NAME},
    )
