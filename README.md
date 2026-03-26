# Physically Accurate Differentiable Inverse Rendering for Radio Frequency Digital Twin

<p align="center">
  <a href="https://witwin.ai">
    <img src="https://img.shields.io/badge/Part%20of-WiTwin.AI-blue?style=for-the-badge" alt="Part of WiTwin.AI"/>
  </a>
</p>

> [!IMPORTANT]
> RFDT uses [**RayD**](https://github.com/Asixa/RayD), our self-developed differentiable ray-tracing core, as its ray-tracing backend.
> If you are looking for the Mitsuba version, please see the [**`mitsuba` branch**](https://github.com/Asixa/Mini-Differentiable-RF-Digital-Twin/tree/mitsuba).

> [!NOTE]
> This is a **conceptual demo** with minimal implementation. The full simulator is coming soon on [**WiTwin.AI**](https://witwin.ai).

### [MobiCom 2026](https://www.sigmobile.org/mobicom/2026/) | [Project Page](https://rfdt.witwin.ai/)

[Xingyu Chen](https://xingyuchen.me/)<sup>1</sup>,
[Xinyu Zhang](http://xyzhang.ucsd.edu/)<sup>1</sup>,
[Kai Zheng](https://kaizheng.me/)<sup>1</sup>,
[Xinmin Fang](https://fangxm.me/)<sup>2</sup>,
[Tzu-Mao Li](https://cseweb.ucsd.edu/~tzli/)<sup>1</sup>,
[Chris Xiaoxuan Lu](https://christopherlu.github.io/)<sup>3</sup>,
[Zhengxiong Li](https://cse.ucdenver.edu/~lizheng/)<sup>2</sup>

<sup>1</sup>University of California, San Diego &nbsp;&nbsp;
<sup>2</sup>University of Colorado Denver &nbsp;&nbsp;
<sup>3</sup>University College London

---

<p align="center">
  <img src="https://rfdt.witwin.ai/static/images/pull_fixed.png" alt="RFDT Teaser" width="90%"/>
</p>

**RFDT** constructs digital twins by solving the inverse problem of RF simulation.
*Top:* 3D reconstruction through radar simulation.
*Bottom:* Communication system optimization through RF scene rendering.

## Abstract

Digital twins (DTs) are virtual replicas of physical scenes that transform the design and evaluation of wireless systems. Previous RF simulators are non-differentiable, limiting their potential for DT construction. We present **RFDT**, a physically-based differentiable RF simulation framework that enables gradient-based interaction between virtual and physical worlds. RFDT overcomes discontinuities in RF ray tracing using a physically grounded edge-diffraction transition function, and mitigates non-convexity from Fourier-domain processing through a signal-domain transform surrogate. Our framework demonstrates the ability to accurately reconstruct digital twins from real RF measurements and supports augmentation of downstream applications including ML-based RF sensing and communication system optimization.

## Differentiating RF Fields w.r.t. Geometric Parameters

<p align="center">
  <img src="https://rfdt.witwin.ai/static/images/grad_param.png" alt="Gradients w.r.t. different scene parameters" width="90%"/>
</p>

**Gradients w.r.t. different scene parameters** (object position, rotation, and Tx position), with finite difference as ground truth. RFDT achieves SSIM up to **0.9997** and PSNR up to **63.09 dB**.

## Taming Non-Convexity with Surrogate Models

<p align="center">
  <img src="https://rfdt.witwin.ai/static/images/surrogate.png" alt="Signal-domain transform surrogate" width="90%"/>
</p>

**Signal-domain transform surrogate.** By replacing FFT-domain representations with smooth surrogates (PSF, Dirichlet), the optimization landscape becomes convex-like, enabling reliable convergence.

## What Does It Enable?

<p align="center">
  <img src="https://rfdt.witwin.ai/static/images/PINN_fixed.png" alt="Enhancing neural networks" width="48%"/>
  &nbsp;
  <img src="https://rfdt.witwin.ai/static/images/whitebox_fixed.png" alt="White-box optimization" width="48%"/>
</p>

- **Enhancing neural networks.** RFDT acts as a physics-informed regularizer that backpropagates through the simulation loop, enabling test-time adaptation of pre-trained models to unseen environments without any labeled data.
- **Replacing neural networks.** With fully differentiable scene parameterization, RFDT directly optimizes geometry, materials, and RF attributes end-to-end — achieving interpretable, physics-grounded solutions without black-box learned components.

## Getting Started

### Runtime Backend

This repository uses [**RayD**](https://github.com/Asixa/RayD) as its ray-tracing backend, **not Mitsuba**.
**RayD** is our self-developed differentiable ray-tracing core, and the current RFDT demo runs on top of **RayD + DrJit**.

### Installation

```bash
conda activate witwin2
pip install -r requirements.txt
```

Requires Python 3.10+, a CUDA-capable GPU, and the **RayD + DrJit** runtime.
The current notebooks and scripts are validated against the `witwin2` conda environment.

### Notebooks

This demo includes 5 Jupyter notebooks that illustrate the core ideas of RFDT:

| Notebook | Description |
|---|---|
| [`forward.ipynb`](forward.ipynb) | **Forward simulation** — Computes the RF field distribution (LoS + reflection + diffraction) around a cube using mesh-based UTD ray tracing. A good starting point to understand the simulation pipeline. |
| [`grad_position.ipynb`](grad_position.ipynb) | **Gradient w.r.t. object position** — Computes ∂field/∂x via automatic differentiation (AD) and validates against finite differences (FD). |
| [`grad_rotation.ipynb`](grad_rotation.ipynb) | **Gradient w.r.t. object rotation** — Computes ∂field/∂θ via AD and validates against FD. |
| [`grad_transmitter.ipynb`](grad_transmitter.ipynb) | **Gradient w.r.t. transmitter position** — Computes ∂field/∂tx via AD and validates against FD. |
| [`optimize.ipynb`](optimize.ipynb) | **Inverse optimization** — Given a target RF field, jointly optimizes object position and rotation to reconstruct the scene. Demonstrates the full inverse rendering pipeline. |

## BibTeX

```bibtex
@inproceedings{chen2026rfdt,
  title     = {Physically Accurate Differentiable Inverse Rendering
               for Radio Frequency Digital Twin},
  author    = {Chen, Xingyu and Zhang, Xinyu and Zheng, Kai and
               Fang, Xinmin and Li, Tzu-Mao and Lu, Chris Xiaoxuan
               and Li, Zhengxiong},
  booktitle = {Proceedings of the 32nd Annual International Conference
               on Mobile Computing and Networking (MobiCom)},
  year      = {2026},
  doi       = {10.1145/3795866.3796686},
  publisher = {ACM},
  address   = {Austin, TX, USA},
}
```
