cd eval/lmms-eval
pip install -e .
pip install -e ".[metrics]"
cd ../..
cd train
pip install -e ".[train]"
cd ../..

pip install --upgrade --force-reinstall --no-cache-dir "numpy<2" scikit-learn pandas