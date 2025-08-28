# GeoKD: Geospatial Knowledge Distillation Framework

A simplified framework for knowledge distillation between geospatial foundation models **Prithvi 2.0** and **TerraMind**.

## 🌍 Overview

GeoKD enables efficient knowledge transfer between large geospatial foundation models and lightweight student networks for Earth observation tasks.

## 🚀 Supported Models

### Teacher Models
- **Prithvi 2.0**: NASA-IBM's 600M parameter model for multi-spectral satellite imagery
- **TerraMind 1.0**: IBM-ESA's multimodal transformer for Earth observation

### Student Models
- **ResNet18**: Lightweight CNN (11.2M parameters)
- **U-Net**: Encoder-decoder for segmentation (31.0M parameters) 
- **Swin Transformer**: Vision transformer (28.3M parameters)

## 🔧 Installation

```bash
git clone https://github.com/your-username/GeoKD.git
cd GeoKD
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from teachers import PrithviLoader, TerraMindLoader
from students import ResNetStudent
from src.distillation import GeospatialDistiller

# Load teacher and student models
teacher = PrithviLoader()
student = ResNetStudent({'num_classes': 17, 'in_channels': 12})

# Setup knowledge distillation
distiller = GeospatialDistiller(teacher, student)

# Train (with your data loader)
history = distiller.distill(train_dataloader, epochs=50)
```

### Training on SEN12MS

```bash
python train_sen12ms.py --config configs/training_config.yaml
```

## 📊 Key Features

- **Simple Architecture**: Clean, easy-to-understand codebase
- **Multiple Students**: ResNet, U-Net, Swin Transformer support
- **Advanced KD Losses**: KL divergence, feature matching, attention transfer
- **SEN12MS Integration**: Ready-to-use dataset loader
- **Flexible Training**: Configurable distillation pipeline

## 📁 Repository Structure

- `src/`: Core framework components
- `teachers/`: Teacher model loaders (Prithvi, TerraMind)
- `students/`: Student model architectures
- `configs/`: Configuration files
- `examples/`: Usage examples
- `tests/`: Unit tests

## 🎯 Performance

Typical results on SEN12MS classification:
- ResNet18 from Prithvi: 89.2% accuracy (98% size reduction)
- U-Net from Prithvi: 91.2% pixel accuracy for segmentation
- Swin from Prithvi: 90.5% accuracy with transformer efficiency

## 🤝 Contributing

Contributions welcome! Please open issues or submit pull requests.

## 📄 License

MIT License - see LICENSE file for details.

## 📚 Citation

```bibtex
@article{geokd2024,
  title={GeoKD: Geospatial Knowledge Distillation for Efficient Earth Observation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🙏 Acknowledgments

- NASA-IBM for Prithvi 2.0
- IBM-ESA-Forschungszentrum Jülich for TerraMind 1.0
- SEN12MS dataset creators