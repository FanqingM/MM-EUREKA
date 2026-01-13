<div align="center">
    <img alt="MM-Eureka logo" src="./docs/logo.png" style="height: 200px;" />
</div>

<div align="center">

# MM-EUREKA

</div>

<hr>
<div align="center">
<p style="text-align: center;">MM-Eureka: Toward Stable Multimodal Reasoning via Rule-based Reinforcement Learning with Policy Drift Control</p>
</div>
<hr>

## 🎯 Overview

**MM-Eureka** addresses a fundamental instability in multimodal rule-based reinforcement learning that causes catastrophic training collapse. We identify that ratio-based policy objectives (e.g., PPO, GRPO, RLOO) can amplify policy drift under sparse multimodal rewards, leading to mid-training collapse where accuracy drops to near-zero and models produce degenerate outputs.

Our solution consists of three key components:

1. **CPGD (Clipped Policy Gradient Optimization with Policy Drift)**: A stability-oriented RL objective that removes ratio-induced amplification while maintaining proximal updates through explicit policy drift regularization and numerically stable KL estimation.

2. **MMK12 Dataset**: A K12-level multimodal reasoning dataset with 15,616 training problems and 2,000 evaluation questions across mathematics, physics, chemistry, and biology, all with human-verified solutions.

3. **MM-Eureka Models**: Trained with CPGD on MMK12, demonstrating stable long-horizon training without collapse. MM-Eureka-7B achieves 10% overall improvement on Qwen2.5-VL-7B, while MM-Eureka-32B reaches competitive performance compared to much larger models.

**Key Insight**: This work focuses on resolving training instability rather than maximizing absolute accuracy. By addressing the structural failure mode in ratio-based objectives, we enable stable multimodal RL training that was previously infeasible.

## 🚀 Features

This repository provides a complete pipeline for stable multimodal reinforcement learning, built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) with the following enhancements:

### Core Algorithm: CPGD
- **Ratio-free gradient term**: Removes importance sampling ratios from the gradient-carrying loss to prevent amplification
- **Explicit policy drift control**: Uses forward KL-based regularization with numerically stable estimation
- **Clipped log-ratio**: Applies clipping on log(π/π_old) instead of raw ratio to prevent explosion
- Enable CPGD with: `--use_cpg_loss --use_policy_drift`
  - `--policy_drift_coef`: Weight of policy drift regularizer (default: 0.01)
  - `--policy_drift_clip_eps`: Clipping range for policy drift (default: 0.2)

### Multimodal RL Infrastructure
- **Vision-Language Model Support**: Extended OpenRLHF to support VLMs (currently InternVL, Qwen2.5-VL)
- **Multiple RL Algorithms**: CPGD, GRPO, RLOO, REINFORCE++ implementations using Ray
- **Distributed Training**: vLLM integration with hybrid engine support (`--colocate_all_models`, `--vllm_enable_sleep`)
- **Rule-based Rewards**: Comprehensive reward visualization (Format Reward, Accuracy Reward, Repetition Penalty)

### Training Stability Features
- **Online Accuracy Filtering**: Filter experiences during training to prevent collapse
  - `--enable_accuracy_filter --accuracy_lower_bound 0.1 --accuracy_upper_bound 0.9`
- **Adaptive Rollout Adjustment (ADORA)**: Dynamic advantage estimation (`--use_adora --adora_lamda`)
- **DAPO Loss**: Optional Direct Advantage Policy Optimization (`--use_dapo`)

## 🤖 Models

We train MM-Eureka models using CPGD on MMK12 dataset, demonstrating stable training throughout the entire process without collapse. Our models show strong performance on multimodal K12-level reasoning tasks.

### Performance Summary

| Model                  | MathVista | MathVerse | MathVision | OlympiadBench | WeMath | MMK12 |
|------------------------|-----------|-----------|------------|---------------|--------|-------|
| Claude3.7-Sonnet       | 66.8      | 52.0      | 41.3       | 48.9          | 72.6   | 55.3  |
| GPT-4o                 | 63.8      | 50.2      | 30.4       | 35.0          | 68.8   | 49.9  |
| o1                     | 73.9      | 57.0      | 60.3       | 68.0          | 98.7   | 73.9  |
| Gemini2-flash          | 70.4      | 59.3      | 41.3       | 51.0          | 71.4   | 65.2  |
| Qwen-2.5-VL-7B         | 68.2      | 47.9      | 25.4       | 20.2          | 62.1   | 53.6  |
| Qwen-2.5-VL-32B        | 74.7      | 49.9      | 40.1       | 30.0          | 69.1   | 66.8  |
| Qwen-2.5-VL-72B        | 74.8      | 57.6      | 38.1       | 40.4          | 72.4   | 70.5  |
| **MM-Eureka-7B**       | 73.0      | 50.3      | 26.9       | 20.1          | 66.1   | 64.5  |
| **MM-Eureka-32B**      | **74.8**  | 56.5      | 34.4       | 35.9          | **73.4** | **72.2** |

**Key Observations:**
- **Stability**: No mid-training collapse across all experiments
- **Improvement**: 10% overall improvement on 7B model; competitive performance on 32B model
- **Generalization**: Training on mathematics improves performance on physics, chemistry, and biology
- **RL vs SFT**: RL generalizes better than supervised methods across diverse reasoning tasks


## 🏁 Getting Started

### 📦 Installation

```bash
# Clone the repository
cd MM-EUREKA
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

### 📂 Data Preparation

**MMK12 Dataset**: A K12-level multimodal reasoning dataset with 15,616 training problems and 2,000 evaluation questions across mathematics, physics, chemistry, and biology. All problems have human-verified solutions.

The dataset will be made available upon paper acceptance.

#### Custom Dataset Format

For custom datasets, format your data as a JSONL file where each entry follows this structure:

```json
{
  "id": "0",
  "message": "[{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file:///path/to/your/image.jpg\"}, {\"type\": \"text\", \"text\": \"Solve this math problem...\"}]}]",
  "answer": "answer that can be parsed and verified"
}
```

### 🌐 Training with CPGD

Before starting training, ensure that paths in the training scripts are correctly set and environment variables like `$MASTER_ADDR` and `$NODE_RANK` are properly configured.

#### Single Node Training

```bash
sh examples/scripts/train_cpgd_qwen_7b_single_node.sh
```

#### Multi-Node Training

```bash
sh examples/scripts/train_cpgd_qwen_7b_multi_node.sh
```

#### Key Training Arguments

- `--use_cpg_loss --use_policy_drift`: Enable CPGD algorithm
- `--policy_drift_coef 0.01`: Control policy drift regularization strength
- `--enable_accuracy_filter`: Enable online filtering to prevent collapse
- `--freeze_prefix visual`: Freeze vision encoder during training
- `--init_kl_coef 0.0`: Disable additional KL penalty (handled by drift regularizer)


## 🎓 Acknowledgements

We acknowledge the outstanding open-source contributions from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [LMM-R1](https://github.com/TideDra/lmm-r1), and [vLLM](https://github.com/vllm-project/vllm). We also extend our gratitude to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [InternVL](https://github.com/OpenGVLab/InternVL), and [QwenVL](https://github.com/QwenLM/Qwen2.5-VL) for their open-source techniques and base models.
