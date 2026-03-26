# PhySense for Sea Temperature

We test [PhySense](https://arxiv.org/pdf/2505.18190) on sea temperature benchmark. The task requires the model to reconstruct the global sea termperature under different sensor placements and generalize to temporal distribution shifts.

<p align="center">
<img src="..\pic\vis_sea.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Visualization of the sea temperature benchmark, reconstruction performance and the optimized sensor placement.
</p>



<p align="center">
<img src="..\pic\res_pipe.png" height = "250" alt="" align=center />
<br><br>
<b>Table 1.</b> Model comparisons of the turbulent flow and sea temperature benchmarks. 
</p>



## Get Started

1. Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data.

The raw data can be found [[here]](https://drive.google.com/drive/folders/1fXe5to8uygfI2iG4AkNsksaAqHKhar9H?usp=sharing).

If you find the button does not work, please directly try this link: https://drive.google.com/drive/folders/1fXe5to8uygfI2iG4AkNsksaAqHKhar9H?usp=sharing

3. Train the base model and the optimized model. 

We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/sensor_flow_train_sea.sh # to train a base model
bash scripts/sensor_flow_train_sea_walk.sh # to train an optimized model given a base model
```

**We provide our trained base model and our optimized model of different sensor numbers** in this link: https://drive.google.com/drive/folders/1ij1EGJzgSQzW59TujopSDIMCsTtre2D8?usp=sharing


You can reproduce our result using the following inference python scripts:
```bash
python sea_infer.py # to evaluate a base model under random sensor placement
python sea_walk_infer.py # to evaluate the optimized model under optimized sensor placement
```

Note: You need to change the corresponding arguments to your own ones.

4. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.
   - Add the model configuration into `./model_dict.py`.
   - Add a script file under folder `./scripts/`, add a new config in the folder `./configs` and change the `model.name` term in the config.