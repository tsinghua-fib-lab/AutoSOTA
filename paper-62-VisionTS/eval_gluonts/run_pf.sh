for ds in electricity solar-energy walmart jena_weather istanbul_traffic turkey_power
do
  python run.py --dataset=${ds} --save_name=pf.csv --periodicity=freq --context_len=2000 --batch_size 512
done
