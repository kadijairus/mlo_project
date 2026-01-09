# mlo_group_project

Machine Learning group 5

# Project Description
### Overall goal of the project
We will be looking at binary classification for medical application. 

We have partially based on pdata preprocessing on https://www.kaggle.com/code/ahmedtronic/ann-breast-cancer, a previous submission for the dataset.

To ensure fluid collaboration, we build a strict pipeline that help quickly integrate each team member across any platform. 

### What data are you going to run on (initially, may change). Describe overall number of samples, size, modality&
We will be using the "Breast Cancer Wisconsin dataset" from Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
The data is spread over 32 columns, of which 1 is the id (irrelevant) and 1 is the classification (B = benign, M = malignant)
8670 samples are included. 

### What models do you expect to use
First version will be using a standard linear artificial neural network (ANN).

## Project structure

The directory structure of the project looks like this:
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
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
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
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Dataset
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)