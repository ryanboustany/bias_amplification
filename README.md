# Bias Amplification

This repository follows the paper **"When Majority Rules, Minority Loses: Bias Amplification of Gradient Descent"**,  
by François Bachoc, Jérôme Bolte, Ryan Boustany, and Jean-Michel Loubes, published at **NeurIPS 2025**.  
📄 [arXiv preprint](https://arxiv.org/pdf/2505.13122)

---

## Overview

This work provides a theoretical and empirical analysis of **bias amplification** during gradient-based training.  
We show that standard optimization dynamics tend to reinforce pre-existing population imbalances,  
creating a *stereotypical gap* between majority and minority subgroups.

---

## Repository Structure

\`\`\`
bias_amplification/
├── src/              # core models, training, and evaluation code
├── notebooks/        # analysis and visualization notebooks
├── results/          # logs, checkpoints, metrics (ignored)
└── README.md
\`\`\`

---

## Reproducibility

1. Install dependencies (Python ≥ 3.8, PyTorch ≥ 2.0)
2. Prepare datasets (e.g., CIFAR-10, EuroSAT)
3. Run experiments:
   \`\`\`bash
   python src/CIFAR2.py \
  --num_runs 10 \
  --lr 8e-3 \
  --network resnet18 \
  --epochs_list 5000 \
  --kappa 90.0 \
  --tau 90.0 \
4. Visualize results with Jupyter notebooks in \`notebooks/\`.

---

## Reference

\`\`\`bibtex
@inproceedings{bachoc2025bias,
  title     = {When Majority Rules, Minority Loses: Bias Amplification of Gradient Descent},
  author    = {Bachoc, François and Bolte, Jérôme and Boustany, Ryan and Loubes, Jean-Michel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
\`\`\`

---
