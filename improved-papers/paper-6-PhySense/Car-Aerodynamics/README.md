# PhySense for Car Aerodynamics

We test [PhySense](https://arxiv.org/pdf/2505.18190) on practical 3D irregular meshes. The car design task requires the model to reconstruct the surface pressure under different driving velocities and yaw angles.

<p align="center">
<img src="..\pic\overview.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Visualization of the car aerodynamics benchmark, reconstruction performance and the optimized sensor placement.
</p>



<p align="center">
<img src="..\pic\res_car.png" height = "250" alt="" align=center />
<br><br>
<b>Table 1.</b> Model comparisons of the car aerodynamics benchmark. 
</p>



## Get Started

1. Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

Note: Additionally, you need to install [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) since this benchamark has irregular data.

2. Prepare Data.

The raw data can be found [[here]](https://drive.google.com/drive/folders/1Xt395uC-RreWQ_bh5v_Bu-AOwV6SRt_w?usp=share_link), which is **the first** to address physics sensing tasks on 3D irregular meshes. We simulate the data using OpenFOAM.

If you find the button does not work, please directly try this link: https://drive.google.com/drive/folders/1Xt395uC-RreWQ_bh5v_Bu-AOwV6SRt_w?usp=share_link

3. Train the base model and the optimized model. 

We provide the experiment scripts under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/ddp_physense_flow_partial.sh # to train a base model
bash scripts/ddp_physense_flow_partial_walk.sh # to train an optimized model given a base model
```

**We provide our trained base model and our optimized model of different sensor numbers** in this link: https://drive.google.com/drive/folders/1kODTzH5n_njQ7lgL-2i2aXbt33UjGKk9?usp=sharing


You can reproduce our result using the following inference scripts:
```bash
bash scripts/ddp_physense_flow_partial_eval.sh # to evaluate a base model under random sensor placement
bash scripts/ddp_physense_flow_partial_eval_walk.sh # to evaluate the optimized model under optimized sensor placement
```

Note: You need to change the arguments `--data_dir` and `--base_model_path` to your own path.

4. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.
   - Add the model configuration into `./model_dict.py`.
   - Add a script file under folder `./scripts/` and change the argument `--cfd_model`.