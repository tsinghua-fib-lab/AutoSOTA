# PhySense (NeurIPS 2025 Oral)

PhySense: Sensor Placement Optimization for Accurate Physics Sensing [[Paper]](https://arxiv.org/pdf/2505.18190)


Physics sensing plays a central role in many scientific and engineering domains, which inherently involves two coupled tasks: reconstructing dense physical fields from sparse observations and optimizing scattered sensor placements to observe maximum information. While deep learning has made rapid advances in sparse-data
reconstruction, **existing methods generally omit optimization of sensor placements, leaving the mutual enhancement between reconstruction and placement on the shelf**. To change this suboptimal practice, we propose PhySense with the following features:

- PhySense is a **synergistic two-stage framework for accurate physics sensing**, which integrates a flow-based generative reconstruction model with a sensor placement optimization strategy through **projected gradient descent to respect spatial constraints**.

- We prove the learning objectives of reconstruction model and sensor placement optimization are **consistent with classical variance-minimization targets**, providing theoretical guarantees.

- PhySense achieves consistent **state-of-the-art reconstruction accuracy** with 49% relative gain across three challenging benchmarks and discovers **informative sensor placements**.


<p align="center">
<table align="center">
<tr>
<td align="center"><img src=".\pic\paradigm.png" height="250" alt=""></td>
<td align="center"><img src=".\pic\rec.png" height="250" alt=""></td>
<td align="center"><img src=".\pic\placement.png" height="250" alt=""></td>
</tr>
<tr>
<td colspan="3" align="center">
<b>Figure 1.</b> <b>Left:</b> Overview of PhySense. <b>Middle:</b> Flow-based reconstructor. <b>Right:</b> Sensor placement optimizer.
</td>
</tr>
</table>
</p>


## Optimized Sensor Placement v.s. Random Sensor Placement

In real-world scenarios, the number of sensors is usually limited due to spatial constraints, power consumption, and environmental restrictions. As a result, allocating the limited sensors to the most informative positions is crucial for high reconstruction accuracy. 

**Even under the identical reconstruction model, the performance varies dramatically with different sensor placements**. Random sensor placement fails to capture important spatial regions, resulting in degraded performance, whereas placements strategically optimized through reconstruction feedback yield significantly enhanced reconstruction accuracy by targeting informative regions, such as side mirrors of a car. 

These results suggest the existence of **a positive feedback loop between reconstruction quality and sensor placement optimization**.

<p align="center">
<img src=".\pic\overview.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 2.</b> Performance comparison under same reconstruction model but different sensor placements.
</p>


## Get Started

1. Prepare the environment. Install Python 3.10. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```


2. Please refer to different folders for detailed experiment instructions.

3. List of experiments:

- Turbulent Flow benchmark: see [./Turbulent-Flow](https://github.com/thuml/PhySense/tree/master/Turbulent-Flow)
- Sea Temperature benchmark: see [./Sea-Temperature](https://github.com/thuml/PhySense/tree/master/Sea-Temperature)
- Car Aerodynamics benchmark: see [./Car-Aerodynamics](https://github.com/thuml/PhySense/tree/master/Car-Aerodynamics)


## Results

PhySense achieves consistent **state-of-the-art reconstruction accuracy with 49% relative gain** across three challenging benchmarks and discovers **informative sensor placements**.

<p align="center">
<img src=".\pic\res_pipe.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Model comparisons of the turbulent flow and sea temperature benchmarks. 
</p>


<p align="center">
<img src=".\pic\res_car.png" height = "250" alt="" align=center />
<br><br>
<b>Table 2.</b> Model comparisons of the car aerodynamics benchmark. 
</p>

## Showcases

<p align="center">
<img src=".\pic\vis_pipe.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization of the turbulent flow benchmark, reconstruction performance and the optimized sensor placement.
</p>


<p align="center">
<img src=".\pic\vis_sea.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization of the sea temperature benchmark, reconstruction performance and the optimized sensor placement.
</p>


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{ma2025PhySense,
  title={PhySense: Sensor Placement Optimization for Accurate Physics Sensing},
  author={Ma, Yuezhou and Wu, Haixu and Zhou, Hang and Weng, Huikun and Wang, Jianmin and Long, Mingsheng},
  booktitle={Advances in neural information processing systems},
  year={2025}
}
```

## Contact

If you have any questions or want to use the code, please contact [mayuezhou20@gmail.com](mailto:mayuezhou20@gmail.com).


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Transolver

https://github.com/OrchardLANL/Senseiver

https://github.com/ermongroup/ddim

https://github.com/gnobitab/RectifiedFlow

