# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
I have built a Logistic Regression model whose purpose is to peform binary classification of people from different country based provided input features.
For model purpose, the target variable is >50K, <=50K. 

## Intended Use
The intended use of this model is for private, public, or goverment agencies to understand the driving factors behind person's income. 

## Training Data
For training purpose, data was divided into 80/20 split, where 80% of the data was used for training the model. Data have been cleaned and processed where numerical features were scaled and categorical features were encoded to meet the model requirements.

## Evaluation Data
To maintain model's reproducibility, evaluation data is passed through the same pipeline as training data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision score, Recall value, and F-score values are logged into to measure model's performance.
In this case, the test results are as follows: Precision score=0.7457886676875957, recall value=0.60875, fbeta score=0.6703372333103923

## Ethical Considerations
Since data used for this project was public census data, no ethical considerations applies here. We should always be cautious about releasing personal information and get approval before releasing data into public. 

## Caveats and Recommendations
More data processing needs to be done, and try advanced algorithms for better model's performance. 
