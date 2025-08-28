# GeoKD: Geospatial Knowledge Distillation Framework

A comprehensive framework for knowledge distillation between geospatial foundation models, focusing on **Prithvi 2.0** and **TerraMind**.

## 🌍 Overview

This repository provides tools and methodologies for knowledge distillation in the geospatial domain, enabling efficient transfer of knowledge between large foundation models for Earth observation tasks.

## 🚀 Supported Models

### Prithvi 2.0
- **Developer**: NASA-IBM
- **Parameters**: 600M
- **Specialty**: Multi-spectral satellite imagery analysis
- **HuggingFace**: [ibm-nasa-geospatial/Prithvi-EO-2.0-600M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M)

### TerraMind 1.0
- **Developer**: IBM-ESA-Forschungszentrum Jülich
- **Architecture**: Dual-scale transformer encoder-decoder
- **Specialty**: Multimodal any-to-any Earth observation generation
- **Training**: 500B tokens from 9M spatiotemporally aligned samples
- **HuggingFace**: [ibm-esa-geospatial/TerraMind-1.0-large](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large)

## 📁 Repository Structure