output=memory_single_service_eta1000.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output;

for((i=6;i<=6;i++));
do
echo $i | tee -a $output;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output;
	python train_single_service.py --indx=$i --atype='memory-leak1_' --eta=1000 | tee -a $output;
done;
echo "-----one service finish-----" | tee -a $output;	
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output;
