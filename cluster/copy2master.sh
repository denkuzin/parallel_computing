#!/usr/bin/env bash

cluster_name = ${1:-"sparktf"}

gcloud compute scp *.py .*sh  ${cluster_name}-m:~/
gcloud compute scp --recurse jobs ${cluster_name}-m:~/

gcloud compute scp .vimrc install_tensorflow.sh ${cluster_name}-m:~/ --zone "us-east1-b"
#gcloud compute ${cluster_name}-m --command "bash ./install_tefsorflow.sh"

echo -e "gcloud compute \"${cluster_name}-m\""
gcloud compute ssh "${cluster_name}-m"
