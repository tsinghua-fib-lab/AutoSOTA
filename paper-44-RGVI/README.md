# RGVI

This is the official PyTorch implementation of our paper:

> **Elevating Flow-Guided Video Inpainting with Reference Generation**, *AAAI 2025*\
> Suhwan Cho, Seoung Wug Oh, Sangyoun Lee, Joon-Young Lee\
> Link: [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/download/32255/34410) [[arXiv]](https://arxiv.org/abs/2412.08975)

<img src="https://github.com/user-attachments/assets/04bd58ff-330e-4e6e-8b5f-402d0afade6d" width=800>


You can also explore other related works at [awesome-video-inpainting](https://github.com/suhwan-cho/awesome-video-inpainting).


## Demo Video
https://github.com/user-attachments/assets/5654ae19-4c8b-458a-872c-9cdf6bedf699


## Abstract
Existing VI approaches face challenges due to the inherent ambiguity between known content propagation and new content generation. 
To address this, we propose a robust VI framework that integrates **a large generative model** to decouple this ambiguity. To further 
improve pixel distribution across frames, we introduce an advanced pixel propagation protocol named **one-shot pulling**. Furthermore, 
we present the **HQVI benchmark**, a dataset specifically designed to evaluate VI performance in diverse and realistic scenarios.


## Setup
1\. Download [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data) from the official website (necessary only for PFCNet training).

2\. For convenience, I also provide the [pre-processed version](https://drive.google.com/file/d/1DcD94HyluwyZgQtN1Y82kM4BHvb2454k/view?usp=drive_link) (resized to 240p).

3\. Download [HQVI](https://drive.google.com/file/d/1eiscxJO2o8qVlHX52sul2Pz8elOwSWUP/view?usp=drive_link) and 
[DAVIS](https://drive.google.com/file/d/1fBdMdhyStDGurVqgIp8_1SMVb0uIRqIH/view?usp=drive_link) to evaluate video object removal performance.

4\. Download [DAVI](https://drive.google.com/file/d/1i-hNki4Jrmiqg6226AIgN1v5PIL1Epm6/view?usp=drive_link) and 
[YTVI](https://drive.google.com/file/d/1MpJCdakKkMZIHg3cqQ2epVxRo0YKCrGY/view?usp=drive_link) to evaluate video restoration performance.

5\. Download [FCNet](https://drive.google.com/file/d/1KNetMWeYVlgrhgguw45uRfoJDmljw5hT/view?usp=drive_link) and [I3D](https://drive.google.com/file/d/1NhI0rDu7BeqNm0oyxS0jHxjQqhfelx8I/view?usp=drive_link) weights and place them in the ``weights/`` directory.




## Running

### Training
Train PFCNet on the YouTube-VOS dataset using the conventional random masking strategy.

PFCNet does not significantly impact RGVI's stability, so you are free to use any custom network of your choice.


### Testing
Run RGVI with:
```
python run.py
```

Verify the following before running:\
✅ Testing dataset selection\
✅ GPU availability and configuration\
✅ Input resolution selection\
✅ Text prompt for generation mode\
✅ Pre-trained model path


Run the evaluation code with:
```
python post.py
```

Verify the following before running:\
✅ Dataset root specification\
✅ Input resolution selection


## Attachments
[Pre-trained model (PFCNet)](https://drive.google.com/file/d/1wknOIQmOnzPtLpN1Ydfb698tUZVOYuKF/view?usp=drive_link)\
[Pre-computed results (STTN)](https://drive.google.com/file/d/1o7UpiQMtoRFDO080JeDM6h27yPsaMRAA/view?usp=drive_link)\
[Pre-computed results (FGVC)](https://drive.google.com/file/d/17mrUlb23Mr4HVaPVDVVaJZmtIq_qEc9V/view?usp=drive_link)\
[Pre-computed results (FuseFormer)](https://drive.google.com/file/d/1wX4phJ2TRt_pGw8xhLLZWDbPtwztw0XZ/view?usp=drive_link)\
[Pre-computed results (E2FGVI)](https://drive.google.com/file/d/1XYDJPqytDok9DwL0kzPJD3GVZhRCRKC3/view?usp=drive_link)\
[Pre-computed results (ProPainter)](https://drive.google.com/file/d/1S_cA788ThoZ6kujB67CdX0YyAwMdnb2N/view?usp=drive_link)\
[Pre-computed results (RGVI)](https://drive.google.com/file/d/1d65KFF91rV8Rvu8io5hVb7yJKOT5Jsbo/view?usp=drive_link)


## Contact
Code and models are only available for non-commercial research purposes.\
For questions or inquiries, feel free to contact:
```
E-mail: suhwanx@gmail.com
```


## License
The files ``evaluator.py``, ``pfcnet.py``, ``post.py``, ``rgvi.py``, ``run.py``, and other related materials 
including model checkpoints and experimental data, are licensed under the
[CC BY-NC License](https://creativecommons.org/licenses/by-nc/4.0/).


The files ``fcnet.py`` and ``i3d.py`` are from [ProPainter](https://github.com/sczhou/ProPainter) and are licensed 
under the [NTU S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE).

