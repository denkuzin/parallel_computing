num_workers=3

declare -a arr
for i in $(seq 0 $(($num_workers - 1)))
do 
arr+=("tf-w-"${i}":2222")
done
list_workers=$(IFS=,; echo "${arr[*]}")

type_machine=$(echo $(echo $(hostname) | awk -F"-" '{print $2}')) #m or w
if [ "$type_machine" == "w" ] 
then
task_index=$(echo $(hostname) | awk -F"-" '{print $3}')
job_name="worker"
else
task_index=0
job_name="ps"
fi

echo
echo --ps_hosts=tf-m:2222
echo --worker_hosts=$list_workers
echo --job_name=$job_name
echo --task_index=$task_index
echo


python item2vec_distributed.py \
     --ps_hosts=tf-m:2222 \
     --worker_hosts=$list_workers \
     --job_name=$job_name \
     --task_index=$task_index
