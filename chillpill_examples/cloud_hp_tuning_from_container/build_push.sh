#!/usr/bin/env bash
# this script builds and pushes a docker image for this example and then uses it to run a cloud ai platform
# hyperparameter tuning job.
#
# you probably don't want to run this script.  you probably want to run run_cloud_tuning_job.py which first creates a
# training job yaml file and then  calls this script
# either way you should set the PROJECT_ID and BUCKET_NAME environmental variables, either in your shell or by editing
# the defaults here

# set these environmental variables to a gcloud project and bucket that you can access
export PROJECT_ID="${PROJECT_ID:-kb-experiment}"
export BUCKET_NAME="${BUCKET_NAME:-kb-dummy-bucket}"

export IMAGE_REPO_NAME="chillpill_examples_container"
export IMAGE_TAG="latest"
export IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}
export REGION=us-central1

echo "IMAGE URI:" $IMAGE_URI

###############################
# build and push docker image #
###############################
# set working direcotry to chillpill root so that docker context includes the whole package
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${THIS_DIR}/../..

THIS_EXAMPLE_DIR=chillpill_examples/cloud_hp_tuning_from_container

if [ "$(uname)" == "Darwin" ]; then
    # Standard mac
    docker build -f ${THIS_EXAMPLE_DIR}/Dockerfile -t $IMAGE_URI .
    docker push $IMAGE_URI
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Goobuntu needs sudo docker calls
    sudo docker build -f ${THIS_EXAMPLE_DIR}/Dockerfile -t $IMAGE_URI .
    sudo docker push $IMAGE_URI
fi
