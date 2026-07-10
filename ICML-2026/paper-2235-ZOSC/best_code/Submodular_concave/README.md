This folder contains the code files used in the paper titled "Solving the Offline and Online Min-Max Problem of Non-smooth Submodular-Concave Functions: A Zeroth-Order Approach".

Link to the paper: "".

In this project, we try to solve the minimax problems considering general possibly non-smooth submodular-concave objective functions using a ZO extra gradient (ZO EG) algorithm with the aim of finding $\epsilon$-saddle points of the cost function.

This folder contains four Python notebook files:

"subM_Conc.ipynb" includes the implementations of the ZO-EG algorithm for the offline and online robust semi-supervised clustering of a dataset sampled from the two-moon dataset.

"image_seg.ipynb" includes the implementations of the ZO-EG algorithm for the offline and online adversarial image segmentation.

"DNN.ipynb" includes the implementations of building a dataset and training a supervised U-net for online adversarial image segmentation.

"DNN2.ipynb" includes the implementations of building a dataset and training a semi-supervised U-net for online adversarial image segmentation.

This folder includes 4 videos:

"online_dumbbell_segmentation_imageio.mp4" shows the online image segmentation using ZO-EG.

"unet_segmentation.mp4" shows the online image segmentation using a supervised U-net.

"unet_segmentation_seedaware2.mp4" shows the online image segmentation using a semi-supervised U-net.

"online_dumbbell_segmentation_imageio_4rho.mp4" shows the online image segmentation using ZO-EG, but with a high adversary.
