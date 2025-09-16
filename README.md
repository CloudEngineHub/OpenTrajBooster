# TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning

### 📝 [Paper](https://arxiv.org/abs/2509.11839) | 🌍 [Project Page](#) | 🤗 [Model](https://huggingface.co/l2aggle/PPTmodel4UnitreeG1) | 🛢️ [Dataset](https://huggingface.co/datasets/l2aggle/Agibot2UnitreeG1Retarget)

## Overview

TrajBooster leverages abundant existing robot manipulation datasets to enhance humanoid whole-body manipulation capabilities. Our approach retargets end-effector trajectories from diverse robots to target humanoids using a specialized retargeting model. We then perform post-pre-training on a pre-trained Vision-Language-Action (VLA) model with this retargeted data, followed by fine-tuning with minimal real-world data. This methodology significantly reduces the burden of human teleoperation while improving action space comprehension and zero-shot skill transfer capabilities.

## 🚀 What's Included

This repository provides the official implementation of TrajBooster, featuring:

- [x] 🤗 **35-hour retargeted dataset**: Unitree G1 whole-body manipulation actions retargeted from Agibot
- [x] 🤗 **Pre-trained model checkpoint**: PPT_model ready for post-training with teleoperation data  
- [x] 🤖 **Hardware deployment**: Complete setup and code for Unitree G1 robot
- [ ] 🕹️ **Teleoperation system**: Real-robot teleoperation implementation and data collection pipeline (coming soon)
- [x] 🧠 **VLA model deployment**: Real-robot deployment implementation for Vision-Language-Action models
- [ ] 📈 **Training scripts**: Retargeting model training code (coming soon)

> **Note**: This repository builds upon our previous work at [Open_WBC](https://github.com/jiachengliu3/WBC_Deploy). If you find this work useful for your research or projects, please consider giving both repositories a ⭐ **star** to support our ongoing open-source contributions to the robotics community!

## 🎯 Key Features

- **Trajectory-Centric Learning**: Novel approach focusing on end-effector trajectory retargeting
- **Cross-Robot Knowledge Transfer**: Leverage data from diverse robot platforms
- **Minimal Real-World Training**: Reduce dependency on expensive human teleoperation
- **Zero-Shot Capabilities**: Enhanced skill transfer to unseen tasks
- **Whole-Body Control**: Full humanoid robot manipulation capabilities

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




