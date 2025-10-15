# LIPIODOGRAM-EDUCORDIAL-programme

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LIPIODOGRAM-EDUCORDIAL-programme** provides end-to-end training pipelines for multiple *time-to-event (survival analysis)* models based on clinical and lipid (lipiodogram) variables.  
It is part of the EDUCordial project for **time-specific mortality risk estimation in primary care**.

---

## 🧩 Overview

This repository contains implementations of four survival models:

| Model | File | Framework |
|--------|------|------------|
| **Cox Proportional Hazards** | `cox.py` | lifelines |
| **Random Survival Forest (RSF)** | `random_forest_survival.py` | scikit-survival |
| **Survival Support Vector Machine (S-SVM)** | `s_svm_survival.py` | scikit-survival |
| **DeepSurv (Deep Neural Survival Network)** | `DeepSurv/train.py` | PyTorch + pycox |



