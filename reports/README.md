# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:



```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [-] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27) (Farnood)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28) (Victor)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28) (Farnood)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30) (Victor checks)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline (eduard)
* [ ] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

5

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s256613, s204475, s256594, 260025, s240118
Kadi Jairus - s256613
Victor G. H. Rasmussen - s204475
Eduard Haiman - s256594
Xiaoyu He - 260025
Farnood Khordepaz -	s240118

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

Currently, we are using no third-party framework that was not covered in the course. We focused on using tools 
recommended and mastering the pipeline they support. 

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed our dependencies using uv. The source of truth is the pyproject.toml file. To ensure every team member 
has a bit-for-bit identical environment, we use a uv.lock file. A new team member can get an exact copy of the 
environment by Git cloning and running ´uv sync´. Additionally, since our binary data and models are not stored in Git, 
the member must run dvc pull after setting up their local service_account_key.json to fetch the processed tensors and 
model checkpoints from Google Cloud Storage. This dual-layered approach (uv for code, DVC for data) ensures a fully 
reproducible pipeline across different machines.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter template we have filled out the tests, .github, data, dockerfiles, models, reports, wandb, configs 
and src folder. We have removed the notebooks folder because we did not use any Jupyter notebooks in our project. 
We have added: (1) .dvc folder to manage our remote connection to Google Cloud Storage, 
(2) scripts folder to hold helper-script to create smaller data files for testing, 
(3) outputs folder for running our experiments, 
(4) src/mlo_group_project/training folder to hold helper classes for training.py, 
(5) src/mlo_group_project/config folder containing Hydra configuration files (config.yaml and hyperparameter configs), and 
(6) gcp folder for Google Cloud Platform deployment artifacts.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used Ruff for both linting and formatting, enforcing PEP 8 compliance with a 120-character line length. For type 
checking, we used mypy with strict settings enabled. We integrated pre-commit hooks to automatically run Ruff (fix, 
format, and lint) on every commit, along with basic checks for trailing whitespace, YAML syntax, and large files. 
These concepts are critical in larger projects because they ensure code consistency across team members, catch bugs 
early (e.g., type errors before runtime), and make the codebase more maintainable and readable. For example, typing 
helps prevent runtime errors by catching type mismatches during development, while consistent formatting reduces merge 
conflicts and cognitive load when reviewing code. It is a good idea to centralise this, as we did, since having 
individual setups for especially formatting will create conflicts more often than not. 

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented around 25 tests. Primarily, we focused on model robustness and data integrity. For model 
tests we verified output shapes across variable batch sizes, ensured the model handles edge-case inputs (zeros, negative 
numbers) without crashing or producing NaNs, and confirmed the model is deterministic in evaluation mode. And for the 
we validated that processed tensors have the correct shape (30 features), are strictly normalized (MinMax scaling 
between 0-1), and that there is no data leakage between training and test sets.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage is currently 53% (tested by running `uv run invoke test`). Even if we achieved 100% code 
coverage, we would not trust the system to be completely error-free. Code coverage only measures which lines of code 
were executed during testing, not whether the logic or the results are correct. Still, it would be worthwhile to run 
more code than not since it helps catch errors early. It is also important to test which files are actually generating 
code coverage reports since all parts of the pipeline can be important to cover. Some parts, however, are more 
high-risk and should be covered. Earlier parts of the pipeline remain the most critical, but ideally all parts should 
be covered. Code coverage can also help highlight what code is actually used, however using packages such as 
"cProfile" is better at isolating parts to optimise. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We early on agreed to use both branches and pull requests. We even enforced rules so pushes cannot be made directly 
on the main branch but only through pull requests (that also need a review). We did this in an attempt to ensure as 
many people as possible are up to date with the project and try to keep unchecked code out of main. Still, this does 
not ensure the code actually works, and it is up to the reviewer to be thorough and in the long run it is better to 
ensure a continuous pipeline that performs automatic linting, building and testing (as we also do).

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC to track changes in our model and data (bcw.csv) by creating a pipeline in dvc.yaml. Using dvc repro 
ensures that if either the preprocessing logic or the underlying data changes, the entire pipeline is consistently 
updated and reproducible.
A second benefit of DVC is that it allows us to store large binary files, such as our PyTorch models, outside of Git. 
We configured Google Cloud Storage as a remote, enabling team members to use dvc pull to sync the project state 
seamlessly across different environments.
Finally, DVC served as a bridge to our automation; our GitHub Actions are configured to trigger an evaluation 
workflow whenever dvc.lock is updated. This ensures that every new version of the data or model is automatically 
validated before deployment, providing a reliable audit trail for our ML experiments.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continuous integration into 4 separate workflows:
1) tests.yaml for running unit tests,
2) linting.yaml for code quality checks,
3) and evaluation.yaml for model evaluation.
4) TODO: answer cloudbuild.yaml 
The tests workflow runs pytest across three operating systems (Ubuntu, Windows, macOS), with caching enabled via 
uv's setup action for faster dependency installation.
The linting workflow runs on push and pull requests to main/master, executing Ruff for code formatting and linting, 
plus mypy for type checking. The evaluation workflow is triggered automatically when dvc.lock changes (indicating a 
new model in the registry), communicating with GCP to pull the latest artifacts via DVC and running our evaluation. 
All workflows use uv for dependency management with locked dependencies (--locked flag) and implement concurrency 
controls to cancel previous runs on new pushes so we don't waste GitHub action minutes running too many actions.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used Hydra for experiment configuration management. Our config files are organized in "src/mlo_group_project/config/", 
with config.yaml as the main file and hyperparameters in hp/basic.yaml. To run an experiment with default settings:
uv run python src/mlo_group_project/train.py 
Experiments can also be run using invoke. List all tasks with 
uv run invoke --list 
To override hyperparameters: 
uv run python src/mlo_group_project/train.py hp.lr=0.01 hp.batch_size=128 hp.epochs=50
with desired parameters. Hydra automatically creates timestamped output directories (outputs/YYYY-MM-DD/HH-MM-SS/) for 
each run, storing logs and checkpoints separately.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Eduard
Using seed

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Xiaoyu
--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Eduard
--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We introduced logging early in the project to help with debugging. 
We used different levels of logging including debug, info, critical and success. By strategically placing log statements
throughout the codebase, we could trace the execution flow and identify where things went wrong. When a bug was 
reported, we would first check the logs to see the sequence of events leading up to the error. This often provided 
clues about the root cause.
More complicated bugs were solved with group: we discussed the problems in chat or in Zoom. Ofter the other team-member
opened the same branch and commited some fixes directly. This way we could share knowledge and help each other.
We introduced pre-commit hooks to automatically run linting and basic tests before every commit, catching potential 
issues early in the development process.
Regarding profiling, we did run a profiling session using cProfile to identify performance bottlenecks.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

Eduard
We used the following GCP services:
1) Cloud Storage (Buckets): Acts as our DVC remote. Used to store our binary files (model), allowing the team to sync 
data states without bloating the Git repository.
2) Artifact Registry: Used to store and manage our Docker images for consistent deployment.
3) Cloud Build with GitHub integration: Used to automate the building and testing of our Docker images whenever we 
pushed changes to our repository.
4) Google Cloud Run: A serverless compute platform used to deploy our model API.
TODO: add more services if used.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

Eduard
--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Victor
--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

Eduard
--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

Eduard
--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

Victor
Simplicity of our model. 
--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We used FastAPI to create a RESTful API for our project. The endpoint accepts POST requests with input data 
and returns data evaluation in JSON format. The endpoint can be used using curl, but it is also accessible at:
https://streamlit-app-934984265576.europe-west1.run.app/
We included automatic API documentation using FastAPI's built-in Swagger UI, allowing users to easily explore and test 
the endpoint. The API was containerized using Docker, ensuring consistent deployment across different environments. 
We also implemented Google Cloud Build to automate the building and testing of our Docker images whenever we pushed 
changes to our repository. This CI/CD pipeline ensured that our API was always up-to-date and reliable.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We did manage to deploy our API using Google Cloud Run both locally and in the cloud. 
First we served the API locally and tested it using curl. 
Then we added added frontend using Streamlit, which enabled users to interact through a web interface.
Later we containerized our FastAPI application with Docker. We created two distinct cloudbuild.yaml files: 
one for the API and one for the Streamlit application. These files instruct Google Cloud Build to build the Docker 
image, push it to Google Artifact Registry, and then deploy it as a service on Cloud Run. 
To invoke the deployed service, users can use browser to access the Streamlit frontend or use curl commands to send 
POST requests. The production URL is provided by Cloud Run and is accessible at: 
https://streamlit-app-934984265576.europe-west1.run.app/

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

Farnood
--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Farnood
--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We used 7.78 GCP credits during the project. The most expensive service was Container Registry Vulnerability Scanning,
which cost $4.16 because every new Docker image push cost. The second expencive service was Cloud Run, which cost 
$3.39 due to frequent testing of the API.
The private projects cost less than $0.50.
Working in the cloud took a lot of time to set up, especially with permissions and access for all team members. 
However, once set up, it provided a scalable and flexible environment for deploying our application. 
It was good to get some free credits and test the environment. In the future we are able to use our personal 
1000 credits a more experienced manner.      

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented a simple frontend for our API using Streamlit. We did this because it allows non-technical users 
to interact with the model and visualize predictions in real-time without needing to use terminal commands or raw 
HTTP requests.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

Eduard
--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:
> 

The biggest challenges in the project were related to Google Cloud. Setting up GCP services and assuring that all 
team members had the correct permissions and access took considerable time. The amount of different Roles was
overwhelming. We overcame these challenges by sharing our experiences of the setup process among team members.
In general our team-work functioned well. We held regular meetings to discuss progress, challenges, and next steps.
It was sometimes difficult to coordinate schedules among all team members and discuss the project
during live Zoom meetings. This was not surprising, as we were a 100% online team. The use of asynchronous communication 
channels like Slack helped us overcome this.
We also faced challenges in selecting which tasks to prioritize within the limited timeframe of the project. We
addressed this by allowing each team member to focus on areas aligned with their  interests, while ensuring that
everyone had a basic understanding of all parts of the project. It helped that we had group members with different
backgrounds and strengths.
--- Xiaoyu ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s256613 (Kadi Jairus) worked on setting up the project including repository, project structure, shared Google 
Cloud project and access; logging and typing, DVC integration; coordinating team meetings and tasks; improving tests 
with student s240118; merging pull requests and helping with merge conflicts.

Student s204475 (Victor G. H. Rasmussen) worked on:
- Adding the API and a Streamlit-based frontend for CSV upload and results display
- Runnable tasks (e.g., uv run invoke preprocess-data/train/evaluate/visualize/serve-api/serve-ui)
- Adding profilling tools (snakeviz) and using it to isolate related performance tweaks
- Good project hygiene (refactored training code (split train.py into helper modules))
- Improving data.py to persist scaler/feature metadata so inference works on new datasets
- Added/maintaining linting and test automation (GitHub Actions)