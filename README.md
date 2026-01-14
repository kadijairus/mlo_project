# Machine Learning Operations in Breast Cancer Aspirate Malignancy Classification

This is the project of group 5 in the course "Machine Learning Operations" at DTU.

# Project Description

### Overall goal of the project

The goals of this project are:
1. Create a machine learning model for binary classification for medical application - detect from tabular data, if an aspirate is malignant or not.
2. Create automated and reproducible ML pipeline to ensure fluid collaboration and adding of team members.

### Data
#### Dataset
The project is based on [Breast Cancer Wisconsin dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) from Kaggle.
#### Number of samples
The dataset consists of 569 samples:
- 357 benign
- 212 malignant

#### Size
The dataset consists of a single csv file of 551 kB.
#### Modality
The data is **tabular data** with 30 features extracted from images of breast aspirates. The data is spread over 32 columns, of which first one is the id (irrelevant) and second one is the classification (B = benign, M = malignant).

The cell nucleus characteristics computed from images of fine needle aspirates (FNA) of breast tissue are:
- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in **30 features**.

### Models

Initially we will use a standard artificial neural network (ANN).

Similar model has been trained on the same dataset and has shown good performance [ANN Breast Cancer model by Ahmed Hafez](https://www.kaggle.com/code/ahmedtronic/ann-breast-cancer). We have also used some of the code of this submission, e.g. for data preprocessing step.

## Project structure

The project uses [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and is based on [Machine Learning Operations template](https://github.com/SkafteNicki/mlops_template).
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── evaluate.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── mlo_project/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
└── tasks.py                  # Project tasks
```
## How to run
### Setup
1. Clone the repository:
    ```bash
   git clone https://github.com/kadijairus/mlo_project.git
   cd mlo_project
   ```
2. Install uv (optional, for running scripts):
   ```bash
   pip install uv
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Add to .env file (create if it doesn't exist):
   ```env
   WANDB_API_KEY=your_wandb_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS="your_google_cloud_service_account_key.json"
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the scripts

Scripts can be run using `uv run invoke`. See a list of available tasks with
   ```bash
    uv run src/mlo_project/data.py
   ```

#### Data Processing
1. Run the data script:
   ```bash
    uv run src/mlo_project/data.py
   ```
2. Ensure, that the processed data is in `data/processed/` folder.

#### Training
1. To train with default configuration run the training script:
   ```bash
   uv run src/mlo_project/train.py
   ```
2. Use Hydra to change configuration options.
3. Use WandB to monitor training.
4. The trained model will be saved in the `models/` folder.

#### Evaluation
1. Run the evaluation script and specify the model path:
    ```bash
    uv run src/mlo_project/evaluate.py models/your_model_file.pt
    ```

### Running the scripts with Docker
1. Build the image:
   docker build -t breast-cancer-train -f dockerfiles/train.dockerfile .
2. Run the training:
   docker run --rm breast-cancer-train
