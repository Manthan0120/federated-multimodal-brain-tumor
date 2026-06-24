# Federated Multimodal Brain Tumor Classification
 
A privacy-preserving federated learning system for brain tumor detection across heterogeneous simulated hospital clients, trained on multimodal medical imaging data (CT and MRI). Built with Flower, PyTorch, and Ray.
 
---
 
## Overview
 
Centralized machine learning on medical data is often infeasible due to patient privacy regulations (HIPAA, GDPR). This project implements a federated learning pipeline where each hospital client trains locally and only shares model weights — never raw patient data.
 
The system simulates 60 heterogeneous hospital clients with realistic non-IID data distributions, trains a shared backbone via FedAvg across 20 communication rounds, and achieves **94.8% peak validation accuracy** with an MLP local head.
 
---
 
## Architecture
 
```
┌─────────────────────────────────────────────────────┐
│                  FL Server (FedAvg)                 │
│         Aggregates backbone weights only            │
└────────────┬──────────────────────────┬─────────────┘
             │  shared backbone weights  │
   ┌─────────▼──────────┐    ┌──────────▼─────────┐
   │   CT Client (x30)  │    │  MRI Client (x30)  │
   │  MobileNetV3 body  │    │  MobileNetV3 body  │
   │  [frozen shared]   │    │  [frozen shared]   │
   │  + local head      │    │  + local head      │
   │  [RF / MLP / DT]   │    │  [RF / MLP / DT]   │
   │  [never shared]    │    │  [never shared]    │
   └────────────────────┘    └────────────────────┘
```
 
### Key design decisions
 
**Split model architecture:** A frozen MobileNetV3-Small backbone (576-dim features) is shared across the federation via FedAvg. Each client trains a private local classifier head (Random Forest, MLP, or Decision Tree) on its own extracted features. Local heads are never shared — preserving data sovereignty.
 
**Non-IID data simulation:** Dirichlet allocation (α=0.2) assigns label proportions per client, simulating realistic hospital data skew. Client sizes follow a log-normal distribution (80–1800 samples) to mirror real-world variation.
 
**Multimodal support:** Clients are assigned either CT or MRI brain scan images (Healthy vs Tumor). The shared backbone generalizes across both modalities without modality-specific fine-tuning.
 
---
 
## Results
 
Trained over 20 federated rounds with 10 clients per round (50% participation), MLP local heads:
 
| Round | Train Accuracy | Val Accuracy |
|-------|---------------|--------------| 
| 1     | 93.68%        | 90.8%        |
| 5     | 95.95%        | 92.0%        |
| 10    | 94.94%        | 94.0%        |
| 15    | 96.37%        | 93.6%        |
| 19    | 96.54%        | **94.8%**    |
| 20    | 95.19%        | 91.2%        |
 
Peak validation accuracy: **94.8%** at round 19. Train-val gap remains stable (~3-5%), indicating no global overfitting despite heterogeneous client data.
 
---
 
## Stack
 
| Component | Technology |
|-----------|-----------|
| Federated learning | Flower (flwr) |
| Distributed compute | Ray |
| Backbone | MobileNetV3-Small (PyTorch / torchvision) |
| Local heads | Scikit-learn (RF, MLP, Decision Tree) |
| Data simulation | Dirichlet non-IID (NumPy) |
| Image preprocessing | PIL, torchvision transforms |
| Config management | YAML |
| Testing | pytest |
 
---
 
## Project Structure
 
```
federated-multimodal-brain-tumor/
├── configs/
│   ├── default.yaml          # FL rounds, client config, head type
│   └── dp.yaml               # Differential privacy config
├── src/
│   ├── data/
│   │   ├── simulate_clients.py   # Dirichlet non-IID client generation
│   │   ├── preprocess.py         # Per-client DataLoaders (CT/MRI)
│   │   ├── download.py           # Dataset download utility
│   │   └── visualize.py          # Data distribution visualization
│   ├── models/
│   │   ├── backbone.py           # Frozen MobileNetV3 feature extractor
│   │   ├── multimodal.py         # Global FL model (backbone + head interface)
│   │   ├── client.py             # Flower NumPyClient implementation
│   │   └── local_heads.py        # RF, MLP, Decision Tree heads
│   ├── fl/
│   │   ├── run_fl.py             # Federated training loop
│   │   └── evaluate_final.py     # Final model evaluation
│   └── server/
│       ├── server.py             # Flower server setup
│       └── strategy.py           # FedAvg + weighted metrics aggregation
├── notebooks/
│   ├── demo_end_to_end.ipynb
│   ├── preprocess_CT_SCAN.ipynb
│   └── preprocess_MRI.ipynb
├── tests/
│   ├── test_local_heads.py
│   └── test_learning_curve_filename.py
├── outputs/
│   └── learning_curves/          # Per-round metrics JSON + PNG
├── main.py                        # Unified CLI entry point
└── requirements.txt
```
 
---
 
## Quick Start
 
```bash
# Install dependencies
pip install -r requirements.txt
 
# Run full pipeline (generate clients → federated training → evaluate)
python main.py all
 
# Run individual stages
python main.py clients   # Generate 60 heterogeneous clients
python main.py fl        # Federated learning (20 rounds)
python main.py eval      # Final evaluation
```
 
### Switch local head type
 
Edit `configs/default.yaml`:
 
```yaml
client:
  local_classifier: mlp          # Options: random_forest | mlp | decision_tree
```
 
---
 
## Dataset
 
Brain tumor CT and MRI images with binary labels: **Healthy** and **Tumor**.  
Download via Kaggle using `src/data/download.py`, then place under `data/raw/Dataset/`.
 
Expected structure:
```
data/raw/Dataset/
├── Brain Tumor CT scan Images/
│   ├── Healthy/
│   └── Tumor/
└── Brain Tumor MRI images/
    ├── Healthy/
    └── Tumor/
```
 
---
 
## Configuration
 
`configs/default.yaml` controls all experiment parameters:
 
```yaml
fl:
  rounds: 20              # Federated communication rounds
  num_clients: 10         # Clients sampled per round
  fraction_fit: 0.5       # Fraction of clients for training
  fraction_evaluate: 1.0  # Fraction for evaluation
 
data:
  dirichlet_alpha: 0.2    # Lower = more heterogeneous (non-IID)
  iid: false
```
 
---
 
## Key Concepts
 
**Federated Averaging (FedAvg):** Global backbone weights are averaged across participating clients each round, weighted by local dataset size. Only backbone parameters travel over the network — local heads remain on-device.
 
**Non-IID heterogeneity:** Real hospital datasets are not uniformly distributed. Dirichlet allocation with α=0.2 creates severe label imbalance across clients, stress-testing the federation's ability to learn a generalizable global model.
 
**Privacy by design:** No raw images or patient labels leave the client. Only gradient-derived weight updates (backbone parameters) are aggregated server-side.
 
---
 
## Future Work
 
- Differential privacy (DP-SGD) integration — config scaffold in `configs/dp.yaml`
- Federated evaluation across unseen hospital distributions
- Extension to multi-class tumor grading (glioma, meningioma, pituitary)
- Communication-efficient aggregation (FedProx, SCAFFOLD)
 
