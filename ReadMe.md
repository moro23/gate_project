# Graph Attention Auto-Encoder for Attributed Graph Embedding

This project implements a **Graph Attention Auto-Encoder (GAE)** for **Attributed Graph Embedding** using `scikit-learn`, `numpy`, and `tensorflow`. The goal of the model is to learn node embeddings by leveraging both graph structure and node attributes, and to reconstruct the graph structure via an auto-encoder architecture. The model employs a **Graph Attention Layer** to assign attention weights to neighbors during the feature aggregation process, improving the node embeddings.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Graph Attention Auto-Encoder Architecture](#graph-attention-auto-encoder-architecture)
- [Usage](#usage)
  - [Data Input](#data-input)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Graph neural networks are powerful tools for learning representations of graph-structured data. This project implements a Graph Attention Auto-Encoder (GAE) that learns node embeddings by applying attention mechanisms on graph neighborhoods. It then attempts to reconstruct the graph's adjacency matrix from these embeddings.

**Key Components:**
- **Graph Attention Layer:** This layer learns attention weights for neighbors, ensuring more important neighbors have a higher influence during aggregation.
- **Auto-Encoder:** The encoder learns to generate node embeddings, while the decoder attempts to reconstruct the graph structure (adjacency matrix) from these embeddings.

This implementation uses:
- `numpy` for matrix manipulations.
- `tensorflow` for the auto-encoder and attention mechanism.
- `scikit-learn` for evaluation, such as clustering or classification tasks on node embeddings.

## Dependencies

Make sure to install the following dependencies:

```bash
pip install tensorflow scikit-learn numpy
