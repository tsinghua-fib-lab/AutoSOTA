---
license: mit
---
---
license: mit
---
the project's GitHub repository: https://github.com/WangYuLin-SEU/KASAL

***

# DSRSTO-dataset

## 1. Dataset Description

The DSRSTO-dataset is a specialized dataset designed to support research on 3D object symmetry. It includes annotations for seven distinct types of symmetries and is composed of 3D models created using 3D CAD software, making it a valuable resource for tasks such as pose estimation, object recognition, and symmetry-based 3D model analysis.

### Key Features:

Learning Resource: This dataset serves as an excellent learning material, helping researchers quickly learn to use the KASAL software and accelerate their pose estimation task development.

## 2. Dataset Structure

The dataset is organized as follows:

Models: 3D models are designed using 3D CAD software such as Solidworks and Blender.

Symmetry Axes: A JSON file for each object containing symmetry axis data, including discrete, and continuous symmetry information.

The JSON file is organized based on the BOP format: https://github.com/thodan/bop_toolkit

## 3. Project Reference

This dataset was created as part of the KASAL (Key-Axis-based Symmetry Axis Localization) Project. 

You can find more details and access the project's GitHub repository here: https://github.com/WangYuLin-SEU/KASAL

## 4. License

MIT License.

## 5. Contributors

Yulin Wang (Southeast University, China)


If you find our work useful, please cite it as follows: 
```bibtex
@ARTICLE{KASAL,
  author = {Wang, Yulin and Luo, Chen},
  title  = {Key-Axis-Based Localization of Symmetry Axes in 3D Objects Utilizing Geometry and Texture}, 
  journal= {IEEE Transactions on Image Processing}, 
  year   = {2024},
  volume = {33},
  pages  = {6720-6733},
  doi    = {10.1109/TIP.2024.3515801}
}
```
