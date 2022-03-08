# SDACD: An End-to-end Supervised Domain Adaptation Framework for Cross-domain Change Detection

> This is a pytorch implementation of our SDACD framework, 2022, version 1.0.



## Introduction

​    Change Detection is a crucial but extremely challenging task of remote sensing image analysis, and it has made great progress due to the rapid development of deep learning. However, most existing deep learning-based change detection methods try to design elaborate and complicated neural networks for powerful feature representations but ignore the universal domain shift induced by time-varying land cover changes including luminance fluctuations and season changes between pre-event and post-event images, thereby producing sub-optimal results. In this paper, we propose **an end-to-end supervised domain adaptation framework for cross-domain change detection**, namely **SDACD**, to effectively alleviate the domain shift between bi-temporal images for better change predictions. Specifically, our SDACD presents collaborative adaptations at **image level** and **feature level**. Image adaptation exploits generative adversarial learning with style-consistency and cycle-consistency constraints to perform cross-domain style transformation, effectively narrowing down the domain gap in a two-side generation fashion. As to feature adaptation, we extract domain irrelevant features to align different feature distributions, further decreasing the domain gap of cross-domain images. To further improve the performance, we combine three image pairs for the final change predictions, including the initial pre-event and post-event image pair and the transformed image pairs in the same style domain. Extensive experiments and analyses on two benchmarks demonstrate the effectiveness and universality of our proposed framework. Notably, our framework pushes several strong baseline models up to new State-Of-The-Art records, achieving 97.34% on the CDD and 92.36% on the WHU building dataset.



## Performance

>  Here gives some examples of change detection results, comparing with existing methods on CDD Dataset in Figure (a), and Figure(b) is the results on WHU Dataset.  

|                             (a)                              |                             (b)                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![比较结果大图 - CDD](E:\work\SDACD\examples\比较结果大图 - CDD.png) | ![比较结果大图 - WHU](E:\work\SDACD\examples\比较结果大图 - WHU.png) |



## How to Run Our Work

### Requirements

- python 3.6+
- Pytorch>=1.7.1

### Installation

Clone this repo:

```bash
git https://github.com/Perfect-You/SDACD
cd SDACD
```

Install [PyTorch](http://pytorch.org/) 1.7.1+ and other dependencies:

```python
pip/conda install pytorch, tqdm, tensorboardX, opencv-python, pillow, numpy, sklearn
```

### Prepare Datasets

> run write_path.py to generate the train.txt, val.txt and test.txt

```cmd
python write_path.py
```

### Train the model

> submit the train.sh；

```shell
sbatch train.sh
```

> the parameters are modified in metedata.json
>
> this is the parameter discription：
>
> --gpu-ids: from 0 to start；
>
> --lr: learning rate, the default learning rate is 5e-4；
>
> --epochs: default epochs is 110；
>
> --batch-size；
>
> --resume continue train from the resume model
>
> --train_txt_path the path of the train data
>
> --val_txt_path the path of the val data
>
> --test_txt_path the path of the test data

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
```

>  DDP is used by default when training

### Test the model

`python eval.py`

### Visualization

`python visualization.py`

