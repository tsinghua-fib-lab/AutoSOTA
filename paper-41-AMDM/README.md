# Auto-Regressive Moving Diffusion Models for Time Series Forecasting

This is the official repo for "Auto-Regressive Moving Diffusion Models for Time Series Forecasting".

## Requirements

1. Install Python >= 3.10.

2. Required dependencies can be installed by: 
   
   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation

The datasets (ETT, Solar Energy and Exchange) can be obtain from (https://github.com/thuml/iTransformer ), and the dataset Stock can be obtained from (https://github.com/Y-debug-sys/Diffusion-TS ). Please put them to the folder `./Data/datasets` in our repository.


### Training & Sampling

For training & Sampling, you can run:

~~~bash
python main.py --config_path ./Config/etth1.yaml
~~~

**Note:** We provide the corresponding `.yaml` files under the folder `./Config` where all possible options can be altered. You may need to change some hyper-parameters in the model for different forecasting scenarios.


## Acknowledgement

We appreciate the following github repos for their valuable codes:

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/Y-debug-sys/Diffusion-TS

https://github.com/thuml/iTransformer

https://github.com/zalandoresearch/pytorch-ts

https://github.com/Hundredl/MG-TSD

https://github.com/paddlepaddle/paddlespatial

https://github.com/amazon-science/unconditional-time-series-diffusion
