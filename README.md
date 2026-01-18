# Reasoning_is_a_Modality

Paper access:

Our best checkpoint c-d-4: https://huggingface.co/lz7fdmu/Reasoning_is_a_Modality/tree/main

environment:
You need to decide the environment based on your best judgement, due to different hardware e.g. B200 cannot run on PyTorch 2.7.0, arm cpu etc. Not guarantee able to work. Use your best judgment to install the environment.

If you're using 8*B200:
```
conda create -n arc python==3.13
conda activate arc
pip install -r requirements.txt
```

prep dataset:
```
python augment_data.py
```

To reproduce our best model c-d-4:
stage 1 pretrain:
```
bash c-4-pretrain_stage_1.sh
```
Depends on the hardware, GPU memory < 150gb needs to reduce the batch size. Different hardware has different speeds, may be very fast (5-10 hrs) or super slow (100hrs+), and compile optimization might take a very long time.
stage 2 pretrain:
```
bash c-d-4-pretrain_stage_2.sh
```

For test time training on ARC1 and 2:

Open file ```c-d-4-ttt-3_ARC1.sh``` and ```c-d-4-ttt-3_ARC2.sh``` to provide the model directory.

```
bash c-d-4-ttt-3_ARC1.sh 
bash c-d-4-ttt-3_ARC2.sh
```
c-d-4-ttt-3_ARC1 (2 attempts) might consume 10-20 hours under 8*B200 GPU
ARC2 is much faster due to 120 < 400.

For analysis:
First, modify the script under the "analysis" directory, provide the folder name as described.
Next,
```
bash analysis/arc_1_vit.sh
```
or any other variation depends on what you want to analyze.


