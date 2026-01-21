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
├── .dvc                      # DVC configuration files
├── .github/                  # Github actions and dependabot
│   └── workflows/
│       ├── evaluation.yaml
│       ├── linting.yaml
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
├── outputs/
├── reports/                  # Reports
│   └── figures/
├── scripts/
├── src/                      # Source code
│   ├── mlo_project/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── wandb/                    # Weights & Biases files
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Project development requirements
└── tasks.py                  # Project tasks
```
## How to run
We use invoke as our primary project CLI to simplify complex commands.
Ensure your environment is set up with uv sync.

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
3. Add to .env file (create if it doesn't exist):
   ```env
   WANDB_API_KEY=your_wandb_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS="your_google_cloud_service_account_key.json"
   ```
4. Activate virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
5. Install the required dependencies:
   ```bash
   uv sync
   ```

### Update data artifacts

We use DVC to version heavy artifacts.
Use these commands to keep your local environment in sync with the cloud registry.

1. Before running any scripts, you must pull the data artifacts tracked by DVC:
   ```bash
   uv run invoke data-pull
   ```
2. After running a successful training and reaching a new "best" model upload data using:
   ```bash
   uv run invoke promote
   ```
3. Pushing data to DVC updates the local dvc.lock file. Commit this to git.
4. See other invoke tasks in `tasks.py` file or run:
   ```bash
   uv run invoke --list
   ```

### Running the standard pipeline

1. Run preprocess and training if data.py or train.py has changed.
   ```bash
   uv run invoke repro
   ```

2. Optional: preprocess and train can be run separately:
   ```bash
   uv run invoke preprocess-data
   uv run invoke train
   ```

3. Promote best model to cloud registry:
   ```bash
   uv run invoke promote
   ```

4. Run evaluation on the test set
   ```bash
   uv run invoke evaluate
   ```

5. Run all tests
   ```bash
   uv run invoke test
   ```

### Monitoring and profiling

We use Hydra for configuration management.
We use WandB to monitor training.
Training progress and model artifacts are automatically logged to Weights & Biases dashboard.

1. Run training with performance profiling enabled:
   ```bash
   uv run invoke train-profile
   ```
2. Visualise results:
   ```bash
   snakeviz reports/train_profile.prof
   ```

### Running the scripts with Docker
1. Build the image:
   ```bash
   docker build -t breast-cancer-train -f dockerfiles/train.dockerfile .
   ```
2. Run the training:
   ```bash
   docker run --rm breast-cancer-train
   ```

### Inference API & User Interface to Evaluate the Model
#### Local Development (via Invoke)
We provide a backend API for programmatic access and a frontend UI for easy user interaction.
1. Start the backend API server:
   ```bash
   uv run invoke serve-api
   ```
2. Start the frontend UI server:
   ```bash
   uv run invoke serve-ui
   ```
3. Access the UI:
Open your browser and go to `http://127.0.0.1:8501/`.
4. Upload csv file with samples under "Upload dataset".
5. Click "Evaluate Dataset".

#### Containerized Deployment (via Docker)
For a consistent environment, we provide a `docker-compose.yml` that orchestrates the API and UI services in parallel.

1. Build and launch the containers:

```
docker compose up --build
```
2. Accessing the services:
* Frontend UI: Navigate to http://localhost:8501.
* Backend API: Available at http://localhost:8000


#### Cloud Production (Google Cloud Platform)
The evaluation services are hosted on **Google Cloud Run** for high availability and scalability.
* Production URL: https://streamlit-app-934984265576.europe-west1.run.app/
> Note: The cloud-hosted UI is configured to communicate directly with the API service via its internal Cloud Run URL. Ensure the API_URL environment variable is correctly set if redeploying.
