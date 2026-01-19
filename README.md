# Reasoning is a Modality

Paper access:

We discovered the qualitative difference between modern AI systems and human intelligence.
Based on biological and cognitive evidence, we modeling human's ability of justifying an action by decoding the corresponding internal mental states.
In contrast, modern AI systems generate behavior and explanations as statistically plausible continuations of an observable trace; we refer to this mechanism of post-hoc rationalization without subjective experience as "hallucination" in the context of this paper.
Thus, "hallucination" is the core component of modern AI systems, AI systems reasoning is purely relying on "hallucination" mechanism.
We successfuly isolate "hallucination" with training, data, model size etc., the only way to mitiage is to cross the mechanistic gap. 
Then we hypothesize that "Reasoning is a Modality": reasoning should exist as a distinct internal channel, a global controller state, that separates from the low-level workspace on which rules are applied.
We developed a new reasoning model mitigated this gap.
See paper for details.

Our best checkpoint c-d-4 "v8_checkpoint_recurrent_4_neighbor_attn_30_epoch_final.pt": https://huggingface.co/lz7fdmu/Reasoning_is_a_Modality/resolve/main/v8_checkpoint_recurrent_4_neighbor_attn_30_epoch_final.pt?download=true

Environment:
You need to decide the environment based on your best judgement, due to different hardware e.g. B200 cannot run on PyTorch 2.7.0, arm cpu etc. Not guarantee able to work. Use your best judgment to install the environment.

For the single node 8*B200 platform:
```
conda create -n arc python==3.13
conda activate arc
pip install -r requirements.txt
```

Prep dataset:
```
python augment_data.py
```

To reproduce our best model c-d-4:
stage 1 pretrain:
```
bash c-4-pretrain_stage_1.sh
```
Depends on the hardware, GPU memory < 150gb needs to reduce the batch size. Different hardware has different speeds, may be very fast (5-10 hrs) or super slow (1000hrs+), and compile optimization might take a very long time.
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


