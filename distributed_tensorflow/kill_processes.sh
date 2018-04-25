num_workers=10
prefix=tf

gcloud compute ssh --zone "us-east1-b" ${prefix}-m --command "pkill -f ./distributed"

for i in $(seq 0 $(($num_workers - 1)))
do
gcloud compute ssh --zone "us-east1-b" ${prefix}-w-${i} --command "pkill -f distributed" &
done

wait
