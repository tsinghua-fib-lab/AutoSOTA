
<p align="center">
  <img src="/kasal/datasets/icon_md.png" alt="">
</p>

<p align="center">
  <a href="https://pypi.org/project/kasal-6d/">
    <img src="https://img.shields.io/pypi/v/kasal-6d" alt="PyPI Version">
  </a>
  <a href="https://pepy.tech/project/kasal-6d">
    <img src="https://static.pepy.tech/badge/kasal-6d" alt="Downloads">
  </a>
  <a href="https://github.com/WangYuLin-SEU/KASAL/releases/">
    <img src="https://img.shields.io/github/downloads/WangYuLin-SEU/KASAL/total?color=green" alt="GitHub Releases Downloads">
  </a>
</p>

# <img src="/kasal/datasets/K8.ico" width="36"> KASAL: Key-Axis-based Symmetry Axis Localization


KASAL is a project designed for determining the symmetry axis orientation and the rotation center of rotationally symmetric objects. When using KASAL, users need to specify one of the eight predefined rotational symmetry types. Based on the selected type, KASAL identifies all symmetry axes on the given object model. Upon completion of symmetry axis localization, KASAL automatically saves the rotational symmetry information in the [BOP format](https://bop.felk.cvut.cz/ "BOP Website"). This structured symmetry information facilitates seamless integration with 6D pose estimation methods that support the BOP format. Moreover, the extracted symmetry data is beneficial for various applications, including 3D reconstruction, object recognition, and related computer vision tasks.

### <img src="/kasal/datasets/K16.png" width="28">  News!
***
- **Sep 2025**: 🎉 KASAL now officially supports **Windows** and **Ubuntu (Linux)** platforms! 
- **Mar 2025**: 🤗 KASAL has been fully **open-sourced** on GitHub and PyPI.
- **Dec 2024**: 📄 The paper "Key-Axis-based Localization of Symmetry Axes in 3D Objects Utilizing Geometry and Texture" is now available at [DOI: 10.1109/TIP.2024.3515801](https://doi.org/10.1109/TIP.2024.3515801).


### <img src="/kasal/datasets/K17.png" width="28">  Datasets
*** 
To identify which objects exhibit rotational symmetry, you can download the DSRSTO dataset provided with KASAL. Additionally, we have utilized KASAL to determine the symmetry axes of objects in the Google Scanned Objects (GSO) and ShapeNet datasets.

Below are the links to these three datasets:

* DSRSTO: https://huggingface.co/datasets/SEU-WYL/DSRSTO-dataset
* GSO: https://huggingface.co/datasets/SEU-WYL/GSO-SAD
* ShapeNet: https://huggingface.co/datasets/SEU-WYL/ShapeNet-SAD


### <img src="/kasal/datasets/K9.png" width="28">  Installation
*** 
* **Platform Support**:  KASAL now supports both **Windows** and **Ubuntu (Linux)** 🎉

>| Platform | Tested Version |
>|----------|----------------|
>| Windows  | Windows 10     | 
>| Ubuntu   | 20.04+ (glibc ≥ 2.31) |
* **Requirements**: Anaconda 3, MeshLab
* **Install via PyPI**

KASAL is now available on **PyPI**, you can install it directly using:
``````
    pip install kasal-6d
``````
* **Manual Installation**

If you want to install KASAL manually, use the following commands:
``````
    conda create -n kasal python=3.10
    conda activate kasal
    pip install -r requirements.txt # Only needed for manual installation
``````
> **⚠️Note**:  
> If you encounter compatibility issues with the default installation, you can install our recommended environment with:
> ```
>     pip install kasal-6d[recommended]
>     # or
>     pip install -r requirements_recommended.txt 
> ```
* **Quick Start**

After installation, you can quickly test KASAL by running the following demo scripts:
``````
    python demo_texture_meshes.py  
    # or    
    python demo_shape_meshes.py
``````
This will launch the KASAL application and process the example dataset.

### <img src="/kasal/datasets/K10.png" width="28">  Rotational Symmetry Types
***
KASAL supports a total of eight rotational symmetry types, which include three continuous rotational symmetries and five discrete rotational symmetries.

<div style="text-align: center;">
  <img src="/kasal/datasets/fig1.png" alt="">
</div>

In KASAL, you can select any rotational symmetry type from the **"Symmetry Type"** dropdown menu and then click **"Cal Current Obj"** to localize the symmetry axes on the object.  

Additionally, for **The n-fold Prismatic Rotational Symmetry** and **The n-fold Pyramidal Rotational Symmetry**, users must specify the order of rotational symmetry, denoted as *n*.

### <img src="/kasal/datasets/K11.png" width="28">  Symmetry Axis Localization Results
***
Given a **regular dodecahedron** and its specified rotational symmetry type, KASAL can accurately determine the **orientations of all symmetry axes** and the **rotation center** on the object model.  

Furthermore, KASAL provides a **visual representation** of these symmetry axes, including their **directions, orders, and the rotation center**.

<div style="text-align: center; ">
  <img src="/kasal/datasets/result-p20-1.png" alt="">
</div>

In the visualization of **symmetry axis directions** and the **rotation center**, KASAL places the **arrow’s starting point at the rotation center** and aligns its **direction with the symmetry axis**.  

For the visualization of **symmetry axis order**, KASAL first generates a **set of transformation matrices** that satisfy the specified rotational symmetry. It then applies these matrices to **recolor the object’s vertices** accordingly.

<div style="text-align: center; ">
  <img src="/kasal/datasets/result-p20-2.png" alt="">
</div>

### <img src="/kasal/datasets/K12.png" width="28">  Batch Processing
*** 
Given a directory path (e.g., `mesh_path`), KASAL will automatically load all 3D model files from the subfolders within this directory. You then need to manually specify each model's **symmetry type**, **order (if applicable)**, and whether it exhibits **texture rotational symmetry**.  

Once the rotational symmetry information for all objects has been determined, you can click **"Cal All Objs"** to perform batch symmetry axis localization for all models.  

Additionally, KASAL automatically saves the specified rotational symmetry information when switching between objects. However, for the **last object in the directory**, please switch back to the previous object to ensure the data is saved.  

If there is only **one object** in the directory, clicking **"Cal Current Obj"** or **"Cal All Objs"** will complete the symmetry axis localization and automatically save the specified symmetry information.

``````
from kasal.app.polyscope_app import app

mesh_path = 'The directory of your 3D model dataset'

app(mesh_path)

``````

### <img src="/kasal/datasets/K13.png" width="28">  Texture Rotational Symmetry
***
In real-world scenarios, most rotationally symmetric objects exhibit **geometric rotational symmetry**, while a smaller number of objects possess **texture rotational symmetry**.  

By default, KASAL employs a **geometry-based symmetry axis localization mode**. If an object exhibits texture rotational symmetry, users need to manually enable the **"ADI-C"** option.


### <img src="/kasal/datasets/K14.png" width="28">  Assisted Localization
***
KASAL performs well for most objects, but it may encounter errors when handling **imperfect or approximately rotationally symmetric objects**.  

If KASAL fails to correctly localize the symmetry axes, you can enable **"show xyz"** and manually select the **x, y, or z axis** that you believe is closest to the **primary key axis**.  

The **primary key axis** refers to the symmetry axis with the **highest order** on the object.

<div style="text-align: center; ">
  <img src="/kasal/datasets/show xyz.png" alt="">
</div>

### <img src="/kasal/datasets/K15.png" width="28">  Citation
***
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

