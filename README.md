# Doodle Classifier & Generator for QuickDraw!

## Classifier
* Train
    1.  Dataset preparation
        ```
        python ./DataUtils/prepare_data.py -h
        ```
    2. Training
        ```
        python main.py -h
        ```
* Evaluation
    1. Prepare evaluation dataset:
        ```
        python ./DataUtils/prepare_data.py -v 0
        ```
    2. Evaluation
        ```
        python evaluate.py -h
        ```

## Generator
* Train
    1. prepare training data in categories.py
        ```
        python download_data.py
        ```
    2. Training
        ```
        python dcgan.py -h
        ```
        or
        ```
        python dccgan.py -h
        ```
    3. Evaluation
        ```
        python ./Evaluation Evaluation.py -h
        ```