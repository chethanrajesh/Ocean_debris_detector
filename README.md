# Ocean Plastic Detection

A machine learning project for detecting and analyzing ocean plastic pollution using satellite imagery.

## Project Structure

```
ocean_plastic/
├── data/                          # Data directory
│   ├── raw/                       # Raw satellite images
│   │   └── satellite_images/
│   ├── processed/                 # Processed data
│   └── annotations/               # Image annotations and labels
├── models/                        # Model storage
│   ├── pretrained/                # Pre-trained model weights
│   └── trained/                   # Trained model checkpoints
├── notebooks/                     # Jupyter notebooks
├── src/                           # Source code
│   ├── config/                    # Configuration files
│   ├── data/                      # Data loading and downloading
│   ├── preprocessing/             # Data preprocessing
│   ├── models/                    # Model architecture and training
│   ├── graph/                     # Graph-based operations
│   └── utils/                     # Utility functions
├── scripts/                       # Run scripts
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd ocean_plastic
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Data Preparation
```bash
python src/data/download_satellite_data.py
```

### Training
```bash
python scripts/train.py
```

### Inference
```bash
python scripts/predict.py --image <path_to_image>
```

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- See `requirements.txt` for all dependencies

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration, please contact the project maintainer.
