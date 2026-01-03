# RayScaleML
**End-to-end distributed machine learning platform using Ray, optimized for Apple Silicon (M-series), with data preprocessing, distributed training, hyperparameter tuning, and model serving.**

---


## Overview

**RayScaleML** is a production-grade, end-to-end machine learning platform designed to demonstrate **Distributed ML System Design**, **Scalable Data Processing**, and **Model Deployment** using **Ray**-optimzed to run efficiently on **Apple Silicon (M-series)** without relying on paid GPUs.

This project mirrors how modern ML infrastructure teams build scalable ML systems in industry, covering the full lifecycle:

- Large-scale data ingestion and preprocessing
- Feature engineering at scale
- Distributed model training (PyTorch, TensorFlow, scikit-learn)
- Hyperparameter optimization
- Model evaluation and tracking
- Online inference using Ray Serve
- Local-first development with cloud portability(AWS)

---

## Why Ray?

Ray provides a **unified distributed computing framework ** that replaces fragmented stacks. RayScaleML intentionally uses **multiple Ray libraries**:

- **Ray Core** - task & actor parallelism
- **Ray Data** - distributed datasets and preprocessing
- **Ray Train** - distributed training abstractions
- **Ray Tune** - scalable hyperparameter optimization
- **Ray Serve**- production model serving


---

## Key Features


- CPU-first design optimized for Apple Silicon
- End-to-end ML pipeline: ingestion -> preprocessing -> training -> HPO -> serving
- Modular, production-ready, and cloud-portable
- Github Actions CI for linting, type checks, and tests
- Portfolio-ready architecture diagrams


---

## Getting Started

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/rayscale-ml.git
cd rayscale-ml




