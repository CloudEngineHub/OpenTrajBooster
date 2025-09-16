# Isaac-GR00T for OpenWBC User Guide

This project is based on NVIDIA's GR00T-N1.5-3B model and incorporates WBC data configuration ([`OpenWBCDataConfig`](./gr00t/experiment/data_config.py)) for G1 mobile manipulation compatibility. Additionally, visualization enhancements have been made to [`eval_policy`](./scripts/eval_policy.py).

For the original documentation, please refer to [`./README_raw_gr00t.md`](./README_raw_gr00t.md)

---

## 🔧 Installation

First, follow the installation instructions in [`./README_raw_gr00t.md`](./README_raw_gr00t.md) for the basic setup.

Then proceed with the following steps:

```bash
cd Isaac-GR00T

# Note: Modify pyproject.toml to remove opencv-python dependency (should already be done in this repository)

# Install dependencies
pip install -e .
```

**Note:** If you encounter the following error during execution:
```bash
AttributeError: module 'cv2' has no attribute 'CV_8U'
```
Please uninstall and reinstall the headless version of OpenCV:
```bash
pip uninstall opencv-python-headless
pip install opencv-python-headless
```

---

## 📁 Data Preparation

### 1. Download Model

Download the pre-trained VLA model to the same directory level as Isaac-GR00T:
```bash
huggingface-cli download --resume-download nvidia/GR00T-N1.5-3B --local-dir ../models/GR00T-N1.5-3B/
```

and you can also download our [Post-Pre-Trained VLA model](https://huggingface.co/l2aggle/PPTmodel4UnitreeG1):



### 2. Download HuggingFace Test Dataset (Optional: for cola picking test)

```bash
# Login to HuggingFace
huggingface-cli login

huggingface-cli download --repo-type dataset --resume-download JimmyPeng02/pick_cola_gr00t4 \
  --cache-dir ../datasets/test/ --local-dir-use-symlinks False
```

---

## 🧠 Model Fine-tuning

### Test Command

For detailed parameters and usage, refer to [`./README_raw_gr00t.md`](./README_raw_gr00t.md):

```bash
python scripts/gr00t_finetune.py --help

python scripts/gr00t_finetune.py \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ./demo_data/robot_sim.PickNPlace \
  --num-gpus 1 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --batch-size 32
```

### Single-Task Full Model Fine-tuning

Results are saved to `./save/` (you may need to create the save folder first):

```bash
python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_mickey_mouse \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_mickey_mouse \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1

python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_pink_fox \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_pink_fox \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1

python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_toy_cat \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_toy_cat \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1

python scripts/gr00t_finetune.py \
  --output-dir ./save/pick_toy_sloth \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ../datasets/multiobj_pick/pick_toy_sloth \
  --num-gpus 1 \
  --batch-size 64 \
  --report-to tensorboard \
  --max_steps 20000 \
  --data-config openwbc_g1
```

### Multi-Task Fine-tuning

Run the following command directly:

```bash
dataset_list=(
"../datasets/multiobj_pick/pick_mickey_mouse"
"../datasets/multiobj_pick/pick_pink_fox"
"../datasets/multiobj_pick/pick_toy_cat"
"../datasets/multiobj_pick/pick_toy_sloth"
)

python scripts/gr00t_finetune.py \
  --base-model-path ../models/GR00T-N1.5-3B \
  --dataset-path ${dataset_list[@]} \
  --num-gpus 1 \
  --output-dir ./save/multiobj_pick_WBC/ \
  --data-config openwbc_g1 \
  --embodiment-tag new_embodiment \
  --batch-size 128 \
  --max_steps 20000 \
  --report-to tensorboard
```

---

## 📊 Dataset Evaluation

```bash
python scripts/eval_policy.py \
  --plot \
  --model_path ./save/test/checkpoint-1000 \
  --dataset-path ../datasets/multiobj_pick/pick_mickey_mouse \
  --embodiment-tag new_embodiment \
  --data-config openwbc_g1 \
  --modality-keys base_motion left_hand right_hand
```

After evaluation, a visualization image `test_fig.png` will be generated in the Isaac-GR00T root directory, displaying the comparison between model predictions and ground truth curves.

---

## 🤖 Real Robot Inference

### Prerequisites
First, launch HomieDeploy.

### Start Image Server
```bash
cd inference_deploys/image_server
python image_server.py
```

### Start Model Server
```bash
python scripts/G1_inference.py \
  --arm=G1_29 \
  --hand=dex3 \
  --model-path models/multask_pick \
  --goal pick_mickey_mouse \
  --frequency 20 \
  --vis \
  --filt
```

**Important Notes for Successful Operation:**
1. The Unitree G1 humanoid's initial pose should not deviate too far from the teleoperation data
2. 20Hz operation frequency yields optimal results