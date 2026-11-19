# AARGO
This repo is the implementation of our paper entitled AARGO: an annotation-aware routing framework for zero-shot protein function prediction. 

## Framework Overview
<img width="4304" height="1606" alt="架构图" src="https://github.com/user-attachments/assets/3b59839f-6eed-46c7-bc4c-bd4f56c491d3" />
The pipeline of AARGO. The  architecture consists of two parallel expert modules and  an annotation-aware routing mechanism: the parametric  backbone module handles every class with at least  one training positive, and the prototype metric module  handles the remaining zero-shot GO terms. Routing  depends only on the training annotation count n  of each GO term, not on the input protein or its  predicted scores, so every prediction is assigned to  exactly one module deterministically. The two experts  share a single hierarchy-aware weighting principle: the  prototype metric module uses it to construct class  representations, and the backbone  uses it to construct classification-head parameters. The two differ only in where the principle  operates. In the prototype metric module it acts in  a constructive representation space that contains no  class-level learnable parameters; in the backbone it acts  on the classification-head parameters as a learnable  soft constraint.

## Environment
To reproduce the environment, please create and activate the Conda environment using the provided `environment.yml` file.
```shell
conda env create -f environment.yml
conda activate env
```

## Dataset
The datasets are available in the TALE paper’s repository at https://github.com/Shen-Lab/TALE or via Baidu Netdisk at https://pan.baidu.com/s/1516dxG23C9Ag0PdYkkJAIw?pwd=3fgf (access code: 3fgf). The latter is recommended, as it includes the preprocessed output files, which simplifies the reproduction process.

## Training
Run the following commands in order to train (both modules). 
```shell
# Clone the Repository
git clone https://github.com/TianyuFeng1110/AARGO.git
cd AARGO

# download the dataset and place the CAFA3 and TALE folders in data_tale.
# modify config/custom.yml and run custom.py(backbone module).
CUDA_VISIBLE_DEVICES=0 nohup python custom.py > ../custom_{Datasets}_{Ontology}.log & tailf ../custom_{Datasets}_{Ontology}.log

# modify config/prototype.yml and run prototype.py(prototype module).
CUDA_VISIBLE_DEVICES=0 nohup python custom.py > ../prototype_{Datasets}_{Ontology}.log & tailf ../prototype_{Datasets}_{Ontology}.log
```

## Evaluation
Run the following commands to perform evaluation.
```shell
# After the training in the above two steps is complete, modify config/eval.yml and run evaluate.py or other evaluation file.
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py > ../evaluate.log & tailf ../evaluate.log
```
