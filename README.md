# AdmissionRate_MLOPS_Pipeline_Project

**Problem Statement / Goal for this project:** This project aims to build a pipeline for the machine learning model. This is the basic pipeline built with the production-level code. I have utilized abstraction methods and object-oriented programming to build a proper pipeline for the admission rate calculation/ prediction based on several other features like GRE scores, university ranking, GPA, etc. The major aim is to learn how to create a pipeline for machine learning projects. In this project I have used the ZenMl to create the pipeline.

![my1stpipeline](https://github.com/ayush01thakur/AdmissionRate_MLOPS_Pipeline_Project/assets/124871122/79f771c4-81b5-4284-8412-dd09f5571676)
Above is the result of the pipeline built, for this MLOPS project.

This pipeline is built using ZenMl. ZenML is an open-source MLOps (Machine Learning Operations) framework for Data Scientists, ML Engineers, and MLOps Developers. It facilitates collaboration in the development of production-ready ML pipelines. ZenML is known for its simplicity, flexibility, and tool-agnostic nature. https://www.zenml.io/

Dataset Source: https://www.kaggle.com/datasets/alisadeghiaghili/university-admissions

**Training Pipeline** 
In this Project, I have just built the Training Pipeline and it goes as follows.

* `_ingest_data_`: This step will ingest the data and create a DataFrame.
* `_clean_data_`: This step will clean the data by removing null values, unwanted columns/ rows/ outliers, and preprocessing, etc.  
* `_model_training_`: This step will train the model.
* `_model_evaluation_`: This step will evaluate the model and save the metrics.

Project Requirements Used:
* python version: 3.10
* python libraries: zenml, sklearn, abc, pandas, and numpy

Instructions to run the Project (Training Pipeline):
* run the run_pipeline.py file after installing and setting up all the libraries in a virtual environment.
* After running the run_pipeline.py type `zenml up` in `cmd`. If you got an error then try executing `zenml up --blocking` command.
* You then will get the link for the ZenMl dashboard. something like (`http://127.0.0.1:8890 `)
* Enter the login id as `default` and log in without any password. This will lead you to the dashboard.
* Now click on the pipelines tab to see your pipeline.

