# Evaluation for saving best.pt
We found that the [evaluation results of different models](../README.md) tested by Megaface and CPLFW dataset have similar tendency. Hence, we select CPLFW dataset for evaluating the model during training process. Before training, you need to prepare dataset by following steps:
1. download CPLFW dataset ([Tips](../README.md)). 
2. crop the CPLFW dataset.
3. create the image list by [create_path.py](./create_path.py).
4. Edit path of CPLFW in [data_conf.yaml](../data_conf.yaml).
