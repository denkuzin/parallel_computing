#!/bin/bash

# deploy N clusters in different zones (to treat with quates limits)

if [ "$1" == "-h" -o "$1" == "--help" ] || [ "$1" != "deploy" -a "$1" != "delete" -a "$1" != "killer"  ] ; then
echo -e "
Usage: bash $0 <command>
where <command>
    deploy
    delete
    killer - kill cluster if there is no jobs on it
"
exit 1
fi

N=(30 10 30 110)
zones=(europe-west1-b  asia-east1-a us-west1-a us-east1-b)
prefix="tf"

#deploy or delete all clusters
if [ "$1" == "deploy" -o  "$1" == "delete" ]; then
 num_of_zones=${#zones[@]}
 num_of_zones=$((num_of_zones-1))
 work_dir=$(pwd)
 cd bdutil
 for i in $(seq 0 ${num_of_zones});
  do
   zone=${zones[$i]}
   num_inst=${N[$i]}

   echo -e "deploy cluster at ${zone}" 
   ./bdutil -f -n $num_inst -z ${zone} -P ${prefix} \
           -b "dkuzin_clust_${zone}" -u custom/.vimrc \
           -u ../run.sh -u ../trainer.py -u ../reader.py \
           -u ../word2vec_distributed.py \
           -u ../install.sh \
           $1 &
  done
 wait
 cd $work_dir
fi

#kill custer if there is no jobs on it
if [ "$1" == 'killer' ] ; then
  num_of_zones=${#zones[@]}
  num_of_zones=$((num_of_zones-1))
  while true ; do
  for i in $(seq 0 ${num_of_zones});
    do
        zone=${zones[$i]}
        num_inst=${N[$i]}
        echo "checking claster in zone: ${zone} ..."
        ( gcloud compute ssh --zone $zone \
                "${prefix}-m" --command "bash -l -c 'hadoop job -list'" | \
                grep -E "^0 jobs currently running" ) \
        && echo "No jobs! Will retry in a couple of minutes" \
        && sleep 120 \
        && ( gcloud compute ssh --zone $zone \
                   "${prefix}-m" --command "bash -l -c 'hadoop job -list'" | \
                   grep -E "^0 jobs currently running" ) \
        && echo "No jobs again! Terminating." \
        && ( cd bdutil && ./bdutil -f -n $num_inst -z ${zone} -P ${prefix} \
           -b "dkuzin_clust_${zone}" delete && cd .. )   
    done

    echo -e "sleeping & retrying\n"
    sleep 60
  done
fi

if [ "$1" == 'deploy' ] ; then
  #print commands to connect to masters
  echo -e "\ncommands to connect ot masters\n"
  for zone in ${zones[*]};
  do
    echo "gcloud compute ssh --zone ${zone} \"${prefix}-m\""
  done

  #open new terminals and SSH to masters
  for zone in ${zones[*]};
  do
    command_text="gcloud compute ssh --zone ${zone} \"${prefix}-m\""
    gnome-terminal -e "$command_text"
  done
fi

if [ "$1" == 'deploy' ] ; then
   #ssh tunnels to masters and open chrome-browsers
   #https://cloud.google.com/dataproc/docs/concepts/cluster-web-interfaces
for zone in ${zones[*]};
  do
    gcloud compute ssh  --zone=${zone} --ssh-flag="-D 1080" \
               --ssh-flag="-N" --ssh-flag="-n" ${prefix}-m &

    /usr/bin/google-chrome \
                --proxy-server="socks5://localhost:1080" \
                --host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost" \
                --user-data-dir=/tmp/${prefix}-m/ \
                "http://${prefix}-m:50030" &
    sleep 1
  done
fi


