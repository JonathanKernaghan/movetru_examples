# movetru_examples
Example models created for binary classification, multiclass classification and regression (predicting a continuous number)

In the notebooks folder, there is one jupyter notebook for each of the above categories, where I have conducted exploratory data analysis on the penguins, iris and tips datasets. 

In the scripts folder, this is one script, developed as a pipeline, for each of the above categories, where I perform data cleaning, feature engineering, model training and evaluation. There is an additional script, where I optimise the binary classification example to improve the evaluation metrics.

In the results folder, there are the evaluation metrics I received when running the four scripts. These might change slightly, either up or down, when you run the scripts, due to randomness in model algorithms and the way data is partitioned.

---

**How to run -**

**Using terminal, ensure you have installed the requirements-**

pip install -r requirements.txt

**Now you can run the scripts**

python 1_binary_classification_example.py run_pipeline

python 2_multiclass_classification_example.py run_pipeline

python 3_regression_example.py run_pipeline

Ensure you are in the correct current directory when running the commands.

---

When looking at the evaluation metrics, bear in mind that higher is better for accuracy and f1 score, while for regression lower is better when looking at the mean absolute error metrics.
