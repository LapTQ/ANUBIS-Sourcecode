# Anubis-benchmark

A comprehensive benchmark suite for skeleton-based action recognition algorithms, featuring multiple state-of-the-art method implementations. This project aims to provide researchers with a unified evaluation platform for comparing different algorithms on skeleton-based action recognition tasks.

## ✨ Features

- 🔥 **14 State-of-the-Art Algorithms**: Integration of the most representative skeleton-based action recognition methods
- 📊 **Unified Evaluation Framework**: Standardized training, testing, and evaluation pipeline
- 🚀 **Efficient Implementation**: Optimized code implementations with GPU acceleration support
- 📈 **Comprehensive Benchmarking**: Support for multiple mainstream datasets
- 🛠️ **Easy to Use**: Detailed documentation and example code

## 🏗️ Project Structure

```
Anubis-benchmark/
├── 2s-AGCN/           
├── BlockGCN/          
├── CTR-GCN/          
├── Decoupling_GCN/     
├── DeGCN/            
├── GCN-NAS/          
├── HDGCN/           
├── Hyperformer/     
├── InfoGCN/          
├── LAGCN/            
├── Motif-stgcn/      
├── MS-G3D/           
├── ShiftGCN/         
├── STGCN/
├── STTFormer/                
├── requirements.txt 
└── README.md        
```

## 🛠️ Requirements

### System Requirements
- **Operating System**: Linux (recommended), Windows, macOS
- **Python**: 3.6 or higher
- **CUDA**: 10.2 or higher (recommended for GPU acceleration)

### Core Dependencies
- **PyTorch**: 1.7.0+
- **torchvision**: 0.8.0+
- **numpy**: 1.19.0+
- **scipy**: 1.5.0+
- **scikit-learn**: 0.23.0+
- **matplotlib**: 3.3.0+
- **tqdm**: 4.50.0+
- **PyYAML**: 5.3.0+

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Anubis-benchmark.git
cd Anubis-benchmark
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n anubis python=3.9
conda activate anubis

# Or using virtualenv
python -m venv anubis_env
source anubis_env/bin/activate  # Linux/macOS
# or anubis_env\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Preparation

Download and process Anubis dataset, the dataset is availiable at https://huggingface.co/datasets/Khat865/Anubis-skeleton



### 5. Run Example
```bash
# Using DeGCN as an example
cd DeGCN
python main.py --config ./config/anubis/anubis.yaml
```

## 🔧 Usage Guide


### Training Models
```bash
# Basic training command
python main.py --config config/anubis/anubis.yaml

### Custom Dataset
To use a custom dataset, you need to:
1. Implement data feeder (`feeders/`)
2. Define graph structure (`graph/`)
3. Update configuration files


## 🤝 Contributing

We welcome community contributions! Please follow these steps:

1. **Fork** this repository
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Create Pull Request**

### Adding New Algorithms

1. Create algorithm folder in project root
2. Implement core files:
   - `main.py`: Main program
   - `model/`: Model definitions
   - `config/`: Configuration files
   - `README.md`: Algorithm description
3. Update main README algorithm list
4. Provide benchmark results

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for classes and functions
- Include type hints where appropriate
- Write unit tests for new features

## 📚 Citation

If you find this benchmark useful in your research, please consider citing:

```bibtex

```



## 🙏 Acknowledgments

 Thanks to all original paper authors for their outstanding contributions to skeleton-based action recognition


