# CdWav2Vec

Cdwav2vec is a multilingual speech model that has been trained using 4 Congolese languages. This model reflects the broadest variety of Congolese languages within the pool of multilingual speech models.  We fine-tuned this model for downsteam ASR for 2 languages and obtain benchmark results on 2 public benchmarks, namely Congolese Swahili (Gamayun audio mini-kit) and the Lingala Read Speech Corpus.

As part of Cdwav2vec, we are creating the biggest corpora accessible to the public for 4 languages belonging to the Niger-Congo B language family.. We also trained state-of-the-art ASR models for 2 congolese languages.

## Benchmarks

Our models are evaluated on two publicly accessible benchmarks, the Gamayun audio mini-kit (Congolese Swahili subset) and the Lingala Read Speech Corpus, and the results are shown below.

| model | Ln | Swc |
|----------|----------|----------|
|Cdwav2vec | - | - |
| Cdwav2vec + LM | - | - |
## Resources
### Download models
| language | Acoustic model | language model |
|----------|----------|----------|
|Lingala | fairseq/hf | KenLM |
| Congolese Swahili | fairseq/hf | KenLM |
### Pretrained Model (* )
| Name | Model checkpoint |
|----------|----------|
|Cdwav2vec Base | fairseq |

(* trained on 4 congolese languages, more details can be found here)
## Pipeline
### Setting up the environment
- Setting up pip environment
- Installing / updating libraries
```
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev ffmpeg

 pip install -r w2v_inference/requirements.txt

 pip install packaging soundfile swifter editdistance omegaconf
```
- Installing Fairseq
```
git clone https://github.com/AI4Bharat/fairseq.git
cd fairseq
pip install --editable ./
```
- Installing KenLM
```

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
bash m_process.sh <path/to/the/root/folder/(mucs)>
```
The would result in creation of manifest folders in each language specific folder which can the be used with fairseq for finetuning.

### Language Modelling (LM)
### Evaluation
### Credits
### Cite
### License
### Contact
### Acknowledgements
