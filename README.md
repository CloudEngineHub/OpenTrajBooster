# TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning

### 📝 [Paper](https://arxiv.org/pdf/2509.11839) | 🌍 [Project Page](https://jiachengliu3.github.io/TrajBooster/) | 🤗 [Model](https://huggingface.co/l2aggle/PPTmodel4UnitreeG1) | 🛢️ [Dataset](https://huggingface.co/datasets/l2aggle/Agibot2UnitreeG1Retarget)

## Overview

TrajBooster leverages abundant existing robot manipulation datasets to enhance humanoid whole-body manipulation capabilities. Our approach retargets end-effector trajectories from diverse robots to target humanoids using a specialized retargeting model. We then perform post-pre-training on a pre-trained Vision-Language-Action (VLA) model with this retargeted data, followed by fine-tuning with minimal real-world data. This methodology significantly reduces the burden of human teleoperation while improving action space comprehension and zero-shot skill transfer capabilities.

## 🚀 What's Included

This repository provides the official implementation of TrajBooster, featuring:

- [x] 🤗 **35-hour retargeted dataset**: Unitree G1 whole-body manipulation actions retargeted from Agibot
- [x] 🤗 **Pre-trained model checkpoint**: PPT_model ready for post-training with teleoperation data  
- [x] 🤖 **Hardware deployment**: Complete setup and code for Unitree G1 robot
- [x] 🕹️ **Teleoperation system**: Real-robot teleoperation implementation and data collection pipeline
- [x] 🧠 **VLA model deployment**: Real-robot deployment implementation for Vision-Language-Action models
- [ ] 📈 **Training scripts**: Retargeting model training code (coming soon)
- [ ] 📋 **Documentation Hub**: Comprehensive installation guides, deployment tutorials, and troubleshooting resources (coming soon)


> **Note**: This repository builds upon our previous work at [OpenWBC](https://github.com/jiachengliu3/WBC_Deploy). If you find this work useful for your research or projects, please consider giving both repositories a ⭐ **star** to support our ongoing open-source contributions to the robotics community!


## 🎯 **Key Features**

- **🎯 Trajectory-Centric Learning**: Revolutionary approach leveraging end-effector trajectory retargeting for precise manipulation control
- **🔄 Cross-Robot Knowledge Transfer**: Seamlessly adapt and transfer skills across diverse robot platforms and morphologies  
- **⚡ Minimal Real-World Training**: Dramatically reduce dependency on expensive human teleoperation data collection
- **🚀 Zero-Shot Capabilities**: Enhanced generalization and skill transfer to previously unseen manipulation tasks
- **🤖 Whole-Body Control**: Complete humanoid robot manipulation with integrated Vision-Language-Action model capabilities

---

## 📋 **Deployment Guide**

This comprehensive guide covers three essential deployment phases:

1. **🕹️ Unitree G1 Teleoperation & Data Collection** - Complete setup and implementation
2. **🎯 Post-Training Pipeline** - Utilizing collected data for VLA model fine-tuning  
3. **🤖 Autonomous Deployment** - Real-robot manipulation using post-trained VLA models

> **💡 Quick Start**: We provide a [pre-trained PPT model](https://huggingface.co/l2aggle/PPTmodel4UnitreeG1) for immediate deployment. Follow the sequential steps below for complete project reproduction.

> **🔬 Advanced Users**: Interested in retargeting model training? Jump directly to [Bonus: Retargeting Model Training](#bonus-retargeting-model-training)

### 🔧 **Troubleshooting Resources**
For deployment issues, you could reference these excellent projects first:
- [OpenWBC](https://github.com/jiachengliu3/WBC_Deploy) - Whole-body control implementation
- [AVP Teleoperation](https://github.com/unitreerobotics/xr_teleoperate) - XR teleoperation framework
- [OpenHomie](https://github.com/InternRobotics/OpenHomie) - Humanoid robot deployment

---

### 🕹️ **Phase 1: Teleoperation & Data Collection**

#### **Project Structure**
```
g1_deploy/
│
├── avp_teleoperation/    # Upper-body control & image transmission
│
├── Hardware/            # Wrist camera hardware specs (optional)
│
└── HomieDeploy/         # Lower-body locomotion control
```

#### **Setup Instructions**

**1. 📷 Wrist Camera Setup (Recommended)**
- **Hardware**: Camera specifications and 3D-printable mount files available in `g1_deploy/Hardware/`
- **Benefits**: Significantly improves VLA depth perception and manipulation accuracy

**2. 🦵 Lower-Body Control Configuration**
- Deploy `g1_deploy/HomieDeploy/` to Unitree G1 onboard computer
- Follow setup instructions in `g1_deploy/HomieDeploy/README.md`
- **Result**: Enable joystick-based teleoperation for locomotion

**3. 🖐️ Upper-Body Control Setup**
- Configure `avp_teleoperation` following `g1_deploy/avp_teleoperation/README.md`
- **Dual Deployment**: Deploy on both local PC (image client) and G1 (image server)

#### **✅ Verification Checklist**
- [ ] **Operator 1**: Real-time first-person robot view in Apple Vision Pro
- [ ] **Operator 1**: Smooth arm and hand control via AVP interface
- [ ] **Operator 2**: Responsive locomotion control (walking, squating)

#### **📊 Data Processing**
Follow setup instructions in `OpenWBC_to_Lerobot/README.md`

Convert collected teleoperation data to LeRobot format:

```bash
python convert_3views_to_lerobot.py \
    --input_dir /path/to/input \
    --output_dir ./lerobot_dataset \
    --dataset_name "YOUR_TASK" \
    --robot_type "g1" \
    --fps 30
```

---

### 🎯 **Phase 2: VLA Model Post-Training**

Utilize your collected and processed teleoperation data for model fine-tuning:

📖 **Detailed Instructions**: `VLA_model/gr00t_modified_for_OpenWBC/README.md`

**Training Pipeline**: Post-train our [PPT Model](https://huggingface.co/l2aggle/PPTmodel4UnitreeG1) with your domain-specific data

---

### 🤖 **Phase 3: Autonomous VLA Deployment**

#### **Step 1: Initialize Image Server**
```bash
# Terminal 1 (on Unitree G1)
cd avp_teleoperate/teleop/image_server
python image_server.py
```

> **🔍 Verification**: Test image stream on local PC with `python image_client.py`, then close before proceeding

#### **Step 2: Lower-Body Control Activation**

**A. ⚠️ CRITICAL - System Reset**
```
Execute: L1+A → L2+R2 → L2+A → L2+B
Expected: Arms hang (L2+A) → Arms down (L2+B)
```

**B. Initialize Robot Control**
```bash
# Terminal 2 (on Unitree G1)
cd unitree_sdk2/build/bin
./g1_control eth0  # or eth1 depending on network configuration
```

**C. Launch Policy Inference**
```bash
# Terminal 3 (on Unitree G1) 
python g1_gym_deploy/scripts/deploy_policy_infer.py
```

**D. Legs Activation**
1. Place robot on ground
2. Press `R2` (robot stands)  
3. Press `R2` again (activate autonomous mode)

> **⚠️ SAFETY NOTICE**: Ensure complete understanding of all system components before deployment. Improper usage may result in hardware damage or safety hazards.


**E. Start VLA Model Server**
```
python scripts/G1_inference.py \
  --arm=G1_29 \
  --hand=dex3 \
  --model-path YOUR_MODEL_PATH \
  --goal YOUR_TASK \
  --frequency 20 \
  --vis \
  --filt
```

## Bonus: Retargeting Model Training

### 🚧 *Coming Soon*

Advanced retargeting model training scripts and comprehensive tutorials will be released shortly. Stay tuned for the complete training pipeline implementation.




## 🔗 Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Dataset** | 35-hour Agibot→UnitreeG1 retargeted data (~30GB) | [🤗 HuggingFace](https://huggingface.co/datasets/l2aggle/Agibot2UnitreeG1Retarget) |
| **Model** | Pre-trained PPT model checkpoint (~6GB) | [🤗 HuggingFace](https://huggingface.co/l2aggle/PPTmodel4UnitreeG1) |
| **Paper** | Full technical details and evaluation | [📝 arXiv](https://arxiv.org/abs/2509.11839) |
| **Base Code** | Underlying deployment framework | [🔗 WBC_Deploy](https://github.com/jiachengliu3/WBC_Deploy) |

<!-- ## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{liu2025trajbooster,
  title={TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning},
  author={Liu, Jiacheng and Ding, Pengxiang and Zhou, Qihang and Wu, Yuxuan and Huang, Da and Peng, Zimian and Xiao, Wei and Zhang, Weinan and Yang, Lixin and Lu, Cewu and Wang, Donglin},
  journal={arXiv preprint arXiv:2509.11839},
  year={2025}
}
``` -->

<!-- ## 🙏 Acknowledgments

We thank the open-source robotics community and all contributors who made this work possible. -->




