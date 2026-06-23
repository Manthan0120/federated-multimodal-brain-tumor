# Federated Multimodal Brain Tumor Classification

Privacy-preserving federated learning system for brain tumor detection 
across heterogeneous medical imaging clients using CT and MRI modalities.

## Architecture
- 60 simulated hospital clients with non-IID data (Dirichlet distribution)
- Shared frozen MobileNetV3 backbone aggregated via FedAvg
- Private local classifier heads per client (RF, MLP, DT) — never shared
- Multimodal support: CT and MRI brain scan images (Healthy vs Tumor)
- Distributed training via Ray and Flower (flwr)

## Key Design Decisions
- Backbone frozen after pretraining — only weights shared across federation
- Local heads remain private, preserving patient data sovereignty
- Dirichlet alpha=0.5 simulates realistic hospital data heterogeneity
- Log-normal client sizes (80–1800 samples) mirror real-world variation

## Stack
Python, PyTorch, Flower (flwr), Ray, MobileNetV3, Scikit-learn, Pandas

## Usage
pip install -r requirements.txt
python main.py all      # generates clients, runs FL, evaluates
python main.py fl       # federated learning only
python main.py eval     # final evaluation only
