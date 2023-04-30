## [Google AI4Code competition](https://www.kaggle.com/competitions/AI4Code). 31st/34th solution

Point-wise ranking approach. Model is based on [CodeT5-base](https://huggingface.co/Salesforce/codet5-base) Encoder with 1024 sequence length.  
[Writeup and discussion](https://www.kaggle.com/competitions/AI4Code/discussion/343762) on Kaggle.

## Requirements

**Hardware**  

Training: 4 cores, 16 GB RAM, TPUv3-8  
Inference: 2 cores, 12 GB RAM, P100  

**Software**  

Ubuntu 18.04  
Python: 3.9.7  
CUDA: 11.2 (for GPU inference)  
cuDNN: 8.1.1 (for GPU inference)  

## Install
In fact I used TF 2.8.0 for training, but newer versions should also be OK

```sh
git clone https://github.com/vecxoz/ai4code
cd ai4code
pip3 install -r requirements.txt
```


## Inference using trained weights

```sh
kaggle competitions download -c AI4Code
kaggle datasets download vecxoz/model-codet5base
kaggle datasets download vecxoz/ai4code-weights

unzip -q AI4Code.zip -d AI4Code
unzip -q model-codet5base.zip -d model-codet5base
unzip -q ai4code-weights.zip -d ai4code-weights

python3 infer.py --data_dir=AI4Code --weight_dir=ai4code-weights --model_dir_or_name=model-codet5base
```

If you use newly trained models for inference, adjust ensemble coefficients according to their performance.  
On Kaggle choose P100 GPU notebook, attach 2 datasets [model-codet5base](https://www.kaggle.com/datasets/vecxoz/model-codet5base) and [ai4code-weights](https://www.kaggle.com/datasets/vecxoz/ai4code-weights), and set paths accordingly.


## Create training data

It takes about 3 hours to create the data on a GCP VM.  
For some reason it may take much longer on Kaggle's latest notebook environment.  

```sh
python3 create_data.py --data_dir=AI4Code --out_dir=ai4code-tfrec
```
There is a [prebuilt dataset](https://www.kaggle.com/datasets/vecxoz/ai4code-tfrec) on Kaggle. You can attach it to your notebook or download:
```sh
kaggle datasets download vecxoz/ai4code-tfrec
unzip -q ai4code-tfrec.zip -d ai4code-tfrec
```

## Train

I trained two first folds (0 and 1) for 20 and 7 full epochs respectively.  
Both were interrupted before full convergence.  
Training time is about 3.5 hours per epoch.  

```sh
python3 train.py --data_tfrec_dir=ai4code-tfrec --initial_fold=0 --final_fold=2
```

On Kaggle choose TPU notebook, attach dataset [ai4code-tfrec](https://www.kaggle.com/datasets/vecxoz/ai4code-tfrec), and set path accordingly.  
Due to Kaggle time limits one needs to train each fold in several separate sessions.

## Acknowledgement
Many thanks to the [TRC program](https://sites.research.google/trc/about/) for TPU resources.


