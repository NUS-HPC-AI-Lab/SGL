<h1 align="center"> <p>SGL</p></h1>

<p align="center">
  <picture>
    <img width="20%" alt="SGL" src="./logo.png">
  </picture>
</p>




The official implementation of "2024 A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for accelerating Large VLMs".

> Wangbo Zhao<sup>1</sup>, Yizeng Han<sup>2</sup>,  Jiasheng Tang<sup>2,3</sup>, Zhikai Li<sup>1</sup>, Yibing Song<sup>2,3</sup>, Kai Wang<sup>1</sup>, Zhangyang Wang<sup>4</sup>, Yang You<sup>1</sup>
>
> <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com/?language=zh), <sup>3</sup>Hupan Lab, <sup>4</sup>[The University of Texas at Austin](https://www.tsinghua.edu.cn/)
>
>  [Paper](https://arxiv.org/abs/2412.03324)

## 💥 Overview
![20241230195723](https://github.com/user-attachments/assets/e244efd4-4136-4402-856f-95e87e33d408)

(a) Small VLM-guided visual token pruning in a large VLM (SGP). We update a global attention map aggregated from all layer of a small VLM. This global attention map is used to rank visual tokens and guide the visual token pruning in a large VLM. 

(b) Aggregation of attention maps in SGP. We aggregate the attention score of visual tokens received from prompt tokens and generated tokens across all heads and layers in the small LM. Higher scores indicate greater significance. 

(c) Inference with Small VLM Early Exiting (SEE). When the early exiting decision score from the small VLM is sufficient, the larger VLM will not be invoked.

## 🔨 Usage


1. Please refer to the documentation of [InternVL](https://github.com/OpenGVLab/InternVL) to set up the environment and prepare the data for evaluation.

2. We take 'bash textvqa2B-26B.sh' as an example, which takes InternVL2-2B as the small model to accelerate the large model InternVL2-26B.








## 🤔 Citation
If you found our work useful, please consider citing us.
```
@article{zhao2024stitch,
  title={A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for accelerating Large VLMs},
  author={Zhao, Wangbo and Han, Yizeng and Tang, Jiasheng and Li, Zhikai and Song, Yibing and Wang, Kai and Wang, Zhangyang and You, Yang},
  journal={arXiv preprint arXiv:2412.03324},
  year={2024}
}
```

## 🙏 Acknowledgement
SGL is built with reference to the code of the following projects: [InternVL](https://github.com/OpenGVLab/InternVL), [FastV](https://github.com/pkunlp-icler/FastV), [QWen2-VL](https://github.com/QwenLM/Qwen2-VL), and [LLaVa-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/).

## ☎️ Contact
🔥🔥🔥 If you are interested in this work and hope to cooperate with us, please drop an email to wangbo.zhao96@gmail.com 🔥🔥🔥

