# CdWav2Vec

Supplementary material to the paper "Leveraging Additional Resources for Improving Automatic Speech Recognition on Congolese Low-Resource Languages".

This repository is meant to supplement the above paper. It contains the curated 2 datasets ( The Lingala Read Speech Corpus and Congolese Speech Radio Corpus) as well as the experimental details which should be sufficient to reproduce the results. For any questions, contact abre.ussen@students.jkuat.ac.ke

## Datasets

Access the datasets [here](https://github.com/ussenuk/CdWav2Vec/tree/master/data_prep_scripts)

## Reproducibility
Cdwav2vec is a multilingual speech model that has been trained using 4 Congolese languages. This model reflects the broadest variety of Congolese languages within the pool of multilingual speech models.  We fine-tuned this model for downsteam ASR for 2 languages and obtain benchmark results on 2 public benchmarks, namely Congolese Swahili (Gamayun audio mini-kit) and the Lingala Read Speech Corpus.

As part of Cdwav2vec, we are creating the biggest corpora accessible to the public for 4 languages belonging to the Niger-Congo B language family.. We also trained state-of-the-art ASR models for 2 congolese languages.

## Models
## Resources
### Download models

### Pretrained Model

| Pretrained model | link |
|----------|----------|
|Cdwav2vec (swc) | [link](https://drive.google.com/file/d/1D_2QOItQjoO9H8T912tIUPf4QKYFDWHf/view?usp=sharing) |
| Cdwav2vec (ln) | [link](https://drive.google.com/file/d/178CKcAo2V0OtkbH3s3cs7W6wXRYwydqJ/view?usp=sharing)|
|Cdwav2vec (kon) | [link](https://drive.google.com/file/d/1rub_rVDhoaJ3mWblSydPcp0CMaN59066/view?usp=sharing) |
| Cdwav2vec (Tshi/lua) | [link](https://drive.google.com/file/d/1s_M5WG3TszgaGRIP--dn332EHzVviIWc/view?usp=sharing)|
| Cdwav2vec* (multi) | [link](https://drive.google.com/file/d/1eNEilSzMqnoY1BMGHoG8CZD7yhhlwNN1/view?usp=sharing)|

(* trained on 4 congolese languages, more details can be found here)

### finetuned Model

#### multilingual
|cd_multi |finetuned on | link |
|----------|----------|----------|
|Cdwav2vec  |(swc)| [link](https://drive.google.com/file/d/1gP42NgwJprouqp5KGtcveNC9RE-DgAuL/view?usp=sharing) |
| Cdwav2vec* (multi) |(ln)| [link](https://drive.google.com/file/d/1vMA8AL9BWt0zb8XiXOQA-skbR9g9V6Uq/view?usp=sharing)|

#### monolingual

|cd_mono |finetuned on | link |
|----------|----------|----------|
| Cdwav2vec  |(swc)| [link](https://drive.google.com/file/d/1Q97Gu8W6gkRl7zM1ZWa3t8qSf5v5Nh7m/view?usp=sharing)|
|Cdwav2vec  |(kon)| [link](https://drive.google.com/file/d/1V5bYsok8syVi-sGcq1NA5XpfqEbPCCYF/view?usp=sharing) |
| Cdwav2vec  |(Tshi/lua)| [link](https://drive.google.com/file/d/1z38VcaACUhqyK0yVB3jp5-uWOuoDuLwn/view?usp=sharing)|
| Cdwav2vec  |ln| [link](https://drive.google.com/file/d/1-ogzY9pp5tAC8yAwgXRL0RuumhT8RTd_/view?usp=sharing)|

### Results on additional test sets
Our models are evaluated on two publicly accessible benchmarks, the Gamayun audio mini-kit (Congolese Swahili subset) and the Lingala Read Speech Corpus, and the results are shown below.
| model | Ln | Swc |
|----------|----------|----------|
|Cdwav2vec (cd_multi) | 21.4 | 13.7 |
| Cdwav2vec (cd_multi) + LM | 18.4 | 14.7 |

## Pipeline
### Setting up the environment
- Setting up pip environment
```
sudo apt-get install python3-pip
sudo apt-get install python3-venv
python3 -m venv [name of your environement]
source [name of your envirnment]/bin/activate
```

- Installing / updating libraries
```
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev ffmpeg

git clone https://github.com/ussenuk/CdWav2Vec.git
cd CdWav2Vec

pip install -r w2v_inference/requirements.txt

cd ..
```
- Installing Fairseq (fairseq==0.12.1)
```
git clone https://github.com/AI4Bharat/fairseq.git
cd fairseq
pip install --editable ./
cd ..
```
- Installing KenLM
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
- Installing flashlight
```
git clone --branch v0.3.2 https://github.com/flashlight/flashlight.git
cd flashlight/bindings/python
export USE_MKL=0
python setup.py install
cd ../../..

```
### Pretraining
#### Data preparation
- Step 1: Downloading Audio Dataset (Unlabeled)

```
bash dw_util.sh <path_to_urls> <data_store_path> <num_of_threads>
```

The ```<data_store_path>``` refers to the location where the data will be downloaded. The ```<num_of_threads>``` can be used to control the parallelization.
- Step 2: Voiced Activity detection

```
python vad.py <data_read_dir> <data_write_dir> <folder_name>
```

The ```<data_read_dir>``` is the root of downloaded files which contain downloaded data in language-named-folders.

The ```<data_write_dir>``` is the location for saving the data after VAD step.

The ```<folder_name>``` refers to the names of language-named-folder for which you want to perform this VAD step.

*The reason why folder_name has been kept as a seperate entity is to allow parallelization because one can process multiple folders simultaneously.
- Step 3: SNR filtering
```
python snr.py <data_path> <folder/language_name>
```
where the ```<data_path>``` refers to the root path containing all the audios in language specific folders. Here it refers to the ```<data_write_dir>``` from the previous step. The ```<folder/language_name>``` refers to name of language_specific folder for which snr_filtering needs to be done. The audio data that is rejected is moved in the folder "snr_rejected", which is created automatically.
- Step 4: Chunking
```
python chunking.py <chunking_path>
```
All the audio files present in the ```<chunking_path>``` will be chunked and saved in the same location. The original files are removed.


##### Or alternatively users can use the one single script process_data.sh to run the entire pipeline
- Usage: bash process_data.sh ```</path/to/download>``` ```<num_of_threads>```
- The ```</path/to/download>``` refers to the location where the data will be downloaded.
- The ```<num_of_threads>``` can be used to control the parallelization.
- Please make sure that the relative path is urls directory is ../urls from the script.
#### Manifest creation
For creating language-wise pretraining manifest
```
python path/to/lang_wise_manifest_creation.py /path/to/wave/files --dest /manifest/path --ext $ext --valid-percent $valid
```
For ```/path/to/wav/files/``` we expect the directory to have one folder per language under the parent directory

In our pretraing, we use a ```--valid-percent``` as ```0.03```
#### Training procedure and code
For pretraining the model we do multi-node training and schedule the runs with slurm.

Following is the invocation script for training Cd-Wav2Vec base starting from Wav2Vec2.0 English base ckeckpoint
```
fairseq-hydra-train \
  task.data=/path/to/manifest/directory \
  common.wandb_project=<wandb project name> \
  task._name=temp_sampled_audio_pretraining \
  +task.sampling_alpha=0.7 \
  common.log_interval=200 \
  common.log_format=tqdm \
  dataset.max_tokens=3000000 \
  common.user_dir=/path/to/custom_task/directory \
  checkpoint.save_dir=/path/to/save/model/checkpoints \
  checkpoint.restore_file=/path/to wav2vec2-english-base/checkpoint.pt \
  +optimization.update_freq='[2]' \
  optimization.clip_norm=0.5 \
  checkpoint.reset_optimizer=true \
  distributed_training.distributed_world_size=<total GPUs> \
  distributed_training.distributed_port=$PORT \
  --config-dir /path/to/configs/directory \
  --config-name wav2vec2_base_librispeech"
  ```
Configs of the models are provided in the configs directory
### Finetuning
#### Data preparation
- Sampling correction (if required for a dataset)

For datasets, that are not sampled uniformly at 16kHz, the user may run the following command to normalize the data first.
```
bash normalize_sr.sh <path/to/the/folder/to/normalize> <ext|wav|mp3>
```
#### Manifest creation

  - Make a new directory and name it (say Gamayun_swc)

  - Download and extract the benchmark data inside Gamayun_swc. The data should be extracted in such a way that each folder inside should contain data for a particular language i.e each language specific folder should contain train, valid and test folder and within them the audio + transcript.txt

Note that the transcript.txt contain entries of the following type
```
<filename1> <transcript1> #just the filename and not the path
<filename2> <transcript2>
<filename3> <transcript3>
<filename4> <transcript4>
...
```
Sample structure of folder tree:
```
Gumaya(or Lingala Read Speech Corpus)
    ├── Lingala
    │   ├── test
    │   │   ├── audio
    │   │   └── transcript.txt
    │   ├── train
    │   │   ├── audio
    │   │   └── transcript.txt
    │   └── valid
    │       ├── audio
    │       └── transcript.txt
    └── Congolese Swahili
        ├── test
        │   ├── audio
        │   └── transcript.txt
        ├── train
        │   ├── audio
        │   └── transcript.txt
        └── valid
            ├── audio
            └── transcript.txt
        .
        .
        .
        .
```
  - Creating the manifest
```
bash m_process.sh <path/to/the/root/folder/(dataset)>
```
The would result in creation of manifest folders in each language specific folder which can the be used with fairseq for finetuning.
#### fine-tuning procedure and code
Following is the invocation script for finetuning CdWav2Vec base on a particular language
```
fairseq-hydra-train \
  task.data=/path/to/finetune/manifest/directory/for/a/particular/language \
  common.wandb_project=<wandb project name> \
  model.w2v_path=/path/to/pretrained/model_base.pt \
  common.log_interval=50 \
  common.log_format=tqdm \
  dataset.max_tokens=1000000 \
  checkpoint.save_dir=/path/to/save/model/fine_tune_checkpoints \
  +optimization.update_freq='[1]' \
  distributed_training.distributed_world_size=<total GPUs> \
  --config-dir /path/to/configs/directory \
  --config-name ai4b_base"
  ```
Configs for both the models are provided in the finetune_configs directory

### Language Modelling (LM)
We train 5-grams Statistical LM using KenLM library.
#### Data preparation
Create lm directory path : lm_data
lm_data folder should contain languages specific folder, each folder having a lexicon and a lm_corpus
#### Training details

Run lm-training: ```bash scripts/train_lm.sh <lm directory path> <lang>```

Ouput will be generate at: <lm directory path>/<lang>.

### Evaluation
  #### Evaluation using fairseq (infer.py)
  ```
python3 w2v_inference/infer/infer.py ${manifest_path} --task audio_finetuning \
--nbest 1 --path ${checkpoint_path} --gen-subset ${valid|test} --results-path ${result_path} --w2l-decoder {viterbi | kenlm} \
--lm-weight 0 --word-score 0 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 5000000 \
--post-process letter
  ```
  #### Evaluating a single file (jupyter notebook)
  Run the following notebooks
  
  - infer_single_file_on_lingala_models.ipynb
  
  - infer_single_file_on_swc_models.ipynb
  
### Preprint

```
@ARTICLE{
  author = {{Kimanuka}, Ussen; {Maina}, Ciira wa; {Buyuk}, Osman},
  title = "{Speech Recognition Datasets for Low-resource Congolese Languages}",
  keywords = {Computer Science - Sound, Computer Science - Computation and Language, Electrical Engineering and Systems Science - Audio and Speech Processing},
  year = 2023,
  month = May,
  archivePrefix = {arXiv},
  adsurl = {[url](http://repository.dkut.ac.ke:8080/xmlui/handle/123456789/7946)}
}
```

### Cite
### License
License under which the data set is made available: CC-BY

It grants unrestricted, irrevocable, royalty-free, worldwide, indefinite rights to use the works in any way, by any user and for any purpose. The only requirement is that the user credits the author and other parties designated to receive attribution and retains copyright and licence notices.

### Contact
abre.ussen@students.jkuat.ac.ke
### Acknowledgements
