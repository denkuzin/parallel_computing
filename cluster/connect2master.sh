#!/usr/bin/env bash

cluster_name = ${1:-"sparktf"}

#1. close port if it busy
kill $(ps aux | grep 'ssh-flag=-D 1080' | grep python | awk '{print $2}')

#2. ssh-tunnel
gcloud compute ssh --zone="us-east1-b" --ssh-flag="-D 1080" --ssh-flag="-N" --ssh-flag="-n" ${cluster_name}-m &

#3. connect ot port
/usr/bin/google-chrome \
               --proxy-server="socks5://localhost:1080" \
               --host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost" \
               --user-data-dir=/tmp/temp_dir/ \
               "http://${cluster_name}-m:8088" &
