airtrafic
==============================

Predict air trafic by routes 

Project Organization
------------

Ce qu'il faut retenir sur notre projet :
1- les données d'une route
Il est demandé dans les consignes du projet d'afficher les données d'une route. Une route est donc le chemin entre un aéroport de départ et un d'arrivée. Tous les aéroports ne sont pas reliés entre eux. Nous avons fait le choix d'utiliser le data frame routes qui ne contient que les routes existantes entre deux aéroports. Nous avions commencé en cours l'application en laissant l'utilisateur choisir l'aéroport de départ et celui d'arrivée, cependant en choisissant les deux indépendemment, nous pouvons être confrontés à une erreur si le trajet n'existe pas. Nous avons donc pris l'initiative d'améliorer le code afin que l'utilisateur choisisse directement une route qui existe afin de ne pas avoir d'erreur dans l'affichage du graphique ou l'entrainement du modèle.

2- la date de début du forecast
Lors du cours en classe nous avions crée ce bouton pour choisir la date, cependant, aucune consigne ne demande de la prendre en compte dans le calcul du forecast. Nous avons donc décidé de supprimer ceci de l'application car ce n'est pas utilisé. 



    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── \_can_be_deleted   <- Trash bin (!! git ignored)
    │
    ├── confidential       <- Confidential documents, data, etc. (!! git ignored)
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── working        <- Working, intermediate data that has been transformed.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │                         Also includes sklearn & pyspark pipelines.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-rvm-initial-data-exploration`.
    │
    ├── production
    │   ├── config         <- YAML files with dependancies between tasks, data catalog and others.
    │   ├── pipelines      <- sklearn & pyspark pipelines.
    │   ├── tasks          <- Luigi tasks.
    │   └── scripts        <- Functions used by Luigi tasks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="http://git.equancy.io/tools/cookiecutter-data-science-project/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
