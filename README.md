![Banner](./im.png)
*Image credit: [Sarah Kane](https://www.ast.cam.ac.uk/people/sarah.kane)*

# A JAX-Accelerated Stellar Stream Generator

StreaMAX is a lightweight, high-performance simulator for stellar streams built with JAX.  
It’s designed to make modeling and inference on stellar streams fast enough for Bayesian analysis and modern data-intensive workflows.

---

## 🚀 Features

- Particle-spray modeling of stellar streams  
- Fast 2D track extraction  
- JAX-accelerated integration for GPU/TPU compatibility  
- Automatic differentiation, can easily compute gradients of the models directly via JAX  
- Modular architecture

---

## 🧩 Models

StreaMAX supports:
- Rapid generation of stellar streams using the particle-spray method  
- Efficient extraction of stream tracks for analysis and inference  

---

## 🧪 Quick Start

See the `quick_start.ipynb` notebook for an example on how to:

- Define potential and stream parameters  
- Generate a stream using the particle-spray method  
- Extract and visualize the resulting track  

---

## ⚙️ Installation

### From PyPI

```bash
pip install StreaMAX
```

### From GitHub (development version)

```bash
git clone https://github.com/David-Chemaly/StreaMAX.git
cd StreaMAX
pip install -e .
```

### Manual dependency installation

```bash
pip install -r requirements.txt
```

---

## 📚 Citation & License

This project is released under the MIT License.  
If you use StreaMAX in your research, please cite and reference the repository:

Chemaly, D. et al. 2025 (in prep.). StreaMAX: A JAX-accelerated stellar stream generator.  
GitHub repository: https://github.com/David-Chemaly/StreaMAX

---

## 🪐 Contributing

Pull requests and feature suggestions are welcome.  
If you encounter bugs or wish to contribute new potentials or integrators, please open an issue on GitHub.

---

**Fast. Differentiable. Modular. JAX-powered.**