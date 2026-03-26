# PhySense for Turbulent Flow

We test [PhySense](https://arxiv.org/pdf/2505.18190) on turbulent flow benchmark. The task requires the model to reconstruct the complex fluid dynamics.

<p align="center">
<img src="..\pic\vis_pipe.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Visualization of the turbulent flow benchmark, reconstruction performance and the optimized sensor placement.
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

The raw data can be found [[here]](https://drive.google.com/drive/folders/1c7CCDWdW7uj8bY8_8CRLf0KD88xo7AQB?usp=sharing).

If you find the button does not work, please directly try this link: https://drive.google.com/drive/folders/1c7CCDWdW7uj8bY8_8CRLf0KD88xo7AQB?usp=sharing

3. Train the base model and the optimized model. 

We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/sensor_flow_train_pipe.sh # to train a base model
bash scripts/sensor_flow_train_pipe_walk.sh # to train an optimized model given a base model
```

**We provide our trained base model and our optimized model of different sensor numbers** in this link: https://drive.google.com/drive/folders/1kGpaXA2Vgx6peHmPMGr0QpeuPbxoJVCS?usp=sharing

You can reproduce our result using the following inference python scripts:
```bash
python pipe_infer.py # to evaluate a base model under random sensor placement
python pipe_walk_infer.py # to evaluate the optimized model under optimized sensor placement
```

Note: You need to change the corresponding arguments to your own ones.

4. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.
   - Add the model configuration into `./model_dict.py`.
   - Add a script file under folder `./scripts/`, add a new config in the folder `./configs` and change the `model.name` term in the config.