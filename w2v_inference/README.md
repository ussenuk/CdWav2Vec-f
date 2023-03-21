# Running Evaluation

### From Source

> Install Linux dependencies: 
```
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
```

> Install KenLM
```
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ..
export KENLM_ROOT=$PWD
cd ..
```

> Install python libraries:  ```pip install -r requirements.txt```

> Install torch from official repo: [PyTorch Official](https://pytorch.org/get-started/locally/)

> Install fairseq: 
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
```
> Install Flashlight: the older version :flashlight==1.0.0

```
git clone --branch v0.3.2 https://github.com/flashlight/flashlight.git
cd flashlight/bindings/python
export USE_MKL=0
python setup.py install
cd ../../..
```

## Data Preparation
- Make dataset and checkpoint directories ```mkdir datasets_test && mkdir checkpoints && mkdir checkpoints/language_model && mkdir checkpoints/acoustic_model```
- Prepare test manifest folder using [fairseq](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) and put the manifest folder inside ```datasets_test``` folder. The ```<data_folder_name>``` must be of the form, ```<lang>_*```, where ```lang``` can be lingala, Congolese Swahili.
- Download/Train fine-tuning and language model checkpoints and put it inside ```model_output/acoustic_model``` and ```lm_data/<lm_folder_name>``` folder respectively. Note: ```<lm_folder_name>``` must contain folder ```<lang>``` with ```lm.binary``` and ```lexicon.lst```.

## Usage
> Run inference: 
```
cd w2v_inference/infer
```
python3 infer.py
```


Note that for decoding with LM, the user must specify KENLM_MODEL (path to lm.binary) and LEXICON (path to lexicon.lst).
Futher one can use the other set of arguements to change the parameters for LM decoding.
