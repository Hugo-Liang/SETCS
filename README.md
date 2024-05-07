# Automatic Code Summarization Using Abbreviation Expansion and Subword Segmentation
This repository contains the datasets and source code for the Semantic Enhanced Transformer for Code Summarization (SETCS) used in our paper entitled "Automatic Code Summarization Using Abbreviation Expansion and Subword Segmentation".


### Data preprocessing(Optional)
The `original (filtered)`, `abbreviation expanded`, `(ULM) tokenized`, and `abbreviation tokenized & (ULM) tokenized` datasets can be found [here](https://drive.google.com/drive/folders/1vjaJGEYyHIVq7ZPq3P_7N_Pc6_q7slS7?usp=sharing).


### Environment Settings
* GPU: Nvidia Tesla P40
* OS: CentOS 7.6

```
git clone https://github.com/Hugo-Liang/SETCS.git
cd SETCS
conda create -n SETCS python=3.8
conda activate SETCS
pip install torch==1.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


### Model Training & Testing




### Get Involved
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports.

 
