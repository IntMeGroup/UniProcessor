## UniProcessor: A Text-induced Unified Low-level Image Processor (ECCV 2024)

[Huiyu Duan](https://scholar.google.com/citations?user=r0bRaCMAAAAJ&hl=en), [Xiongkuo Min](https://scholar.google.com/citations?user=91sjuWIAAAAJ&hl=en&oi=ao), Sijing Wu, [Wei Shen](https://scholar.google.com/citations?user=Ae2kRCEAAAAJ&hl=en&oi=ao), and [Guangtao Zhai](https://scholar.google.com/citations?user=E6zbSYgAAAAJ&hl=en&oi=ao)

This is the official repo of the paper [UniProcessor: A Text-induced Unified Low-level Image Processor](https://www.arxiv.org/abs/2407.20928):

<hr />

> **Abstract:** *Image processing, including image restoration, image enhancement, etc., involves generating a high-quality clean image from a degraded input. Deep learning-based methods have shown superior performance for various image processing tasks in terms of single-task conditions. However, they require to train separate models for different degradations and levels, which limits the generalization abilities of these models and restricts their applications in real-world. In this paper, we propose a text-induced unified image processor for low-level vision tasks, termed UniProcessor, which can effectively process various degradation types and levels, and support multimodal control. Specifically, our UniProcessor encodes degradation-specific information with the subject prompt and process degradations with the manipulation prompt. These context control features are injected into the UniProcessor backbone via cross-attention to control the processing procedure. For automatic subject-prompt generation, we further build a vision-language model for general-purpose low-level degradation perception via instruction tuning techniques. Our UniProcessor covers 30 degradation types, and extensive experiments demonstrate that our UniProcessor can well process these degradations without additional training or tuning and outperforms other competing methods. Moreover, with the help of degradation-aware context control, our UniProcessor first shows the ability to individually handle a single distortion in an image with multiple degradations.* 
<hr />


### Installation
See [README_install.md](README_install.md) for the installation of dependencies required to run UniProcessor.

All required downloaded resources can be downloaded here: [UniProcessor](https://pan.sjtu.edu.cn/web/share/e7b32bfc1e25885ac19107bf7315368e), using password: `eqln`.
Or can be downloaded one-by-one using the links below.

### Data preparation
The used datasets can be downloaded from: [datasets](https://pan.sjtu.edu.cn/web/share/7abe8c86a5fc534842eefb926c36e4b8), using password: `cn8k`.

### Pre-trained models
Our trained model can be downloaded from: [model](https://pan.sjtu.edu.cn/web/share/6bc57c1b6b3e43f0ea9d4f42c654417a), using password: `hicw`.

### Results
The results can be downloaded from: [results](https://pan.sjtu.edu.cn/web/share/d2170a442839e1c0d7678036800e693b), using password: `ln4h`.

### Training
See [README_train.md](README_train.md) for the training codes of UniProcessor.

### Testing
See [README_test.md](README_test.md) for the testing codes of UniProcessor.

### Citation
If you use UniProcessor, please consider citing:
```
@inproceedings{duan2024uniprocessor,
  title={UniProcessor: A Text-induced Unified Low-level Image Processor},
  author={Duan, Huiyu and Min, Xiongkuo and Wu, Sijing and Shen, Wei and Zhai, Guangtao},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### Acknowledgements
A large portion of codes in this repo is based on [PromptIR](https://github.com/va1shn9v/PromptIR) and [LAVIS](https://github.com/salesforce/LAVIS).

### Contact
If you have any question, please contact huiyuduan@sjtu.edu.cn
