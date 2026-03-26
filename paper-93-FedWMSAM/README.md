## FedWMSAM

**FedWMSAM (Federated Weighted Momentum with Sharpness-Aware Minimization)** is a novel federated optimization algorithm implemented on top of the [FL-simulator](https://github.com/woodenchild95/FL-Simulator) framework. It is specifically designed to address non-IID challenges in federated learning, thereby improving the model's generalization capability.

## ✨ Highlights

- ✅ Built upon **FL-simulator**, fully compatible with its interface
- ⚖️ Incorporates **Weighted Momentum** to stabilize training under long-tail data distribution
- 📈 Leverages **momentum-driven Sharpness-Aware Minimization (SAM)** to improve model generalization on heterogeneous data

---

## 🚀 Quick Start


```bash
cd NeurlPS_FedWMSAM

# Create a new environment
conda create -n fedwmsam python=3.8
conda activate fedwmsam

# Install dependencies
bash install.sh

# Train ResNet-18 on Cifar-10 with FedWMSAM
# IID
python train.py
# Non-IID Dirichlet 0.1
python train.py --split-rule Dirichlet --split-coef 0.1
# Non-IID Pathological 2
python train.py --split-rule Pathological --split-coef 2