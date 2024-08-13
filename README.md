# Investigating Instruction Tuning Large Language Models on Graphs

[![Paper](https://img.shields.io/badge/Paper-Arvix%20Link-green)](https://arxiv.org/abs/2408.05457)

### Overview

![research_questions](fig/research_questions.svg)

This study delves into the capabilities of instruction-following LLMs for engaging with real-world graphs. In this work, we (1) create a benchmark with fine-grained tasks from two different domain networks, (2) investigate the influence of different graph representations in graph instruction tuning and (3) propose three levels of generalization for graph-related tasks to investigate the generalization of the instruction-tuned LLMs.

Our experiments show that JSON format gives the LLMs the best performance after tuning and LLM can derive algorithms from the learned algorithms.

### Run Experiments

#### Requirements

```bash
pip install -r requirements.txt
```

#### Prepare data

The datasets created in this work can be downloaded from [here](https://drive.google.com/drive/folders/1S2VIMGiiYcqSxPoxoog3THFB2DW2GeIi?usp=sharing). Download the datasets and place them under their corresponding ```data``` directory under ```data_process``` directory.

#### Training

To graph instruction tune the model, go to ```sh``` directory and run the following command

```bash
bash train.sh ../config/train_json.sh
```

#### Testing

To test the graph instruction tuned model or other instruction tuned models, run the following command

```bash
bash eval.sh ../config/test_json.sh # Graph instruction tuned model
bash eval.sh ../config/test_baselines.sh # Other instruction tuned models
```

### Citation
```
@misc{zhu2024investigatinginstructiontuninglarge,
      title={Investigating Instruction Tuning Large Language Models on Graphs}, 
      author={Kerui Zhu and Bo-Wei Huang and Bowen Jin and Yizhu Jiao and Ming Zhong and Kevin Chang and Shou-De Lin and Jiawei Han},
      year={2024},
      eprint={2408.05457},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.05457}, 
}
```