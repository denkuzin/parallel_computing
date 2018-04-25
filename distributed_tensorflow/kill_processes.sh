num_workers=10
prefix=tf

gcloud compute --project "imp-anomaly-research" ssh --zone "us-east1-b" ${prefix}-m --command "pkill -f ./item2vec_distributed"

for i in $(seq 0 $(($num_workers - 1)))
do
gcloud compute --project "imp-anomaly-research" ssh --zone "us-east1-b" ${prefix}-w-${i} --command "pkill -f item2vec_distributed" &
done

wait
