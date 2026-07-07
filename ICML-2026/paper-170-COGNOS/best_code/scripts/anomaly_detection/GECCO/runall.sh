export CUDA_VISIBLE_DEVICES=0

sh 'TimeMixer++.sh'
sh Autoformer.sh
sh CrossAD.sh
sh DLinear.sh
sh KANAD.sh
sh LSTMAE.sh
sh MICN.sh
sh ModernTCN.sh
sh TimesNet.sh
