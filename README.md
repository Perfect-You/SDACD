# SDACD: An End-to-end Supervised Domain Adaptation Framework for Cross-domain Change Detection

This software implements SDACD: An End-to-end Supervised Domain Adaptation Framework for Cross-domain Change Detection in PyTorch. For more details, please refer to our paper https://arxiv.org/abs/2204.00154



## Abstract

​    Change Detection is a crucial but extremely challenging task of remote sensing image analysis, and much progress has been made with the rapid development of deep learning. However, most existing deep learning-based change detection methods try to elaborately design complicated neural networks with powerful feature representations, but ignore the universal domain shift induced by time-varying land cover changes, including luminance fluctuations and season changes between pre-event and post-event images, thereby producing sub-optimal results. In this paper, we propose an end-to-end Supervised Domain Adaptation framework for cross-domain Change Detection, namely SDACD, to effectively alleviate the domain shift between bi-temporal images for better change predictions. Specifically, our SDACD presents collaborative adaptations from both image and feature perspectives with supervised learning. Image adaptation exploits generative adversarial learning with cycle-consistency constraints to perform cross-domain style transformation, effectively narrowing the domain gap in a two-side generation fashion. As to feature adaptation, we extract domain-invariant features to align different feature distributions in the feature space, which could further reduce the domain gap of cross-domain images. To further improve the performance, we combine three types of bi-temporal images for the final change prediction, including the initial input bi-temporal images and two generated bi-temporal images from the pre-event and post-event domains. Extensive experiments and analyses on two benchmarks demonstrate the effectiveness and universality of our proposed framework. Notably, our framework pushes several representative baseline models up to new State-Of-The-Art records, achieving 97.34% and 92.36% on the CDD and WHU building datasets, respectively.

![CD_v1.9](.\examples\CD_v1.9.png)

## Installation

Install [PyTorch](http://pytorch.org/) 1.7.1+ and other dependencies:

```
pip/conda install pytorch>=1.7.1, tqdm, tensorboardX, opencv-python, pillow, numpy, sklearn
```

## Run demo

Generate the train.txt, val.txt and test.txt

```
python write_path.py
```

A demo program can be found in demo. Before running the demo, download our pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1y4GRIUWXh8eNvsy93Z2Smg) (Extraction code: eu68). Set the path of files  in tmp/***.pt. Then launch demo by:

```
python eval.py
```

## Evaluatioin

```
python eval.py
```

```
python visualization.py
```

## Train a new model

Generate the train.txt, val.txt and test.txt:

```
python write_path.py
```

Submit the train.sh:

```
sbatch train.sh
```

## Results

>  Here gives some examples of change detection results, comparing with existing methods on CDD Dataset in Figure (a), and Figure(b) is the results on WHU Dataset.  

|            (a)             |            (b)             |
| :------------------------: | :------------------------: |
| ![CDD](.\examples\CDD.png) | ![WHU](.\examples\WHU.png) |

Evaluation of SDACD on different datasets with SNUNet, STANet, and DASNet as baseline:

| **Methods**  |                          |           CDD            |                          |                           |       WHU building       |                          |
| :----------: | :----------------------: | :----------------------: | :----------------------: | :-----------------------: | :----------------------: | :----------------------: |
|              |         **P(%)**         |         **R(%)**         |         **F(%)**         |         **P(%)**          |         **R(%)**         |         **F(%)**         |
|    FC-EF     |          84.68           |          65.13           |          73.63           |           80.75           |          67.29           |          73.40           |
| FC-Siam-diff |          87.57           |          66.69           |          75.07           |           48.84           |          88.96           |          63.06           |
| FC-Siam-conc |          88.81           |          62.20           |          73.16           |           54.20           |          81.34           |          65.05           |
|    STANet    |          83.17           |          92.76           |          87.70           |           82.12           |          89.19           |          83.40           |
| SDACD-STANet |  87.40   **↑****4.23**   |   89.50  **↓****3.26**   |   88.40  **↑****0.70**   |   90.90  **↑****8.78**    | **93.50**  **↑****4.31** |   92.21  **↑****8.81**   |
|    DASNet    |          93.28           |          89.91           |          91.57           |           83.77           |          91.02           |          87.24           |
| SDACD-DASNet |   92.85  **↓****0.43**   |   91.87  **↑****1.96**   |   92.35  **↑****0.78**   |   89.21  **↑****5.44**    |   90.46  **↓****0.56**   |   89.83  **↑****2.59**   |
|    SNUNet    |          96.60           |          94.77           |          95.68           |           82.12           |          89.19           |          85.51           |
| SDACD-SNUNet | **97.13**  **↑****0.53** | **97.56**  **↑****2.79** | **97.34**  **↑****1.66** | **93.85**  **↑****11.73** |   90.91  **↑****1.72**   | **92.36**  **↑****6.85** |

The grid search results of λf and λCD. Here we fixed λcyc=10 and λi=1.

| Baseline |  λf  | λCD  | P(%)  | R(%)  |   F(%)    |
| :------: | :--: | :--: | :---: | :---: | :-------: |
|          |  1   | 0.05 | 93.38 | 90.85 |   92.10   |
|          |  1   | 0.1  | 93.85 | 90.91 | **92.36** |
|  SNUNet  |  1   | 0.2  | 93.92 | 90.94 |   92.09   |
|          | 0.5  | 0.1  | 93.31 | 91.28 |   92.28   |
|          |  1   | 0.1  | 93.85 | 90.91 | **92.36** |
|          |  2   | 0.1  | 94.56 | 89.99 |   92.22   |

## Acknowledgements

The authors would like to thank the developers of PyTorch, SNUNet, STANet, and DASNet. 
Please let me know if you encounter any issues.

