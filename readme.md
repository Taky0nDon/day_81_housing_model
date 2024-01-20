## Predict Housing Prices
### Relevant Characteristics
	* The number of rooms
	* Distance to employment centres
	* How rich or poor the area is
	* How many students there are per teacher in local schools etc

## Goals

### 1. Analyze and explore the Boston house price data
### 2. Split data for training and testing
### 3. Run a Multivariable Regression
### 4. Evaluate how your model's coefficients and residuals (???)
### 5. Use data transformation to improve model performance
### 6. Use model to estimate a property price

Imports: "scikit-learn" as "import sklearn"
```python
import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# TODO: Add missing import statements
```
## Data Characteristics
>:Number of Instances: 506 
>
>k:Number of Attributes: 13 numeric/categorical predictive. The Median Value (attribute 14) is the target.
>
>:Attribute Information (in order):
>    1. CRIM     per capita crime rate by town
>    2. ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
>    3. INDUS    proportion of non-retail business acres per town
>    4. CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
>    5. NOX      nitric oxides concentration (parts per 10 million)
>    6. RM       average number of rooms per dwelling
>    7. AGE      proportion of owner-occupied units built prior to 1940
>    8. DIS      weighted distances to five Boston employment centres
>    9. RAD      index of accessibility to radial highways
>    10. TAX      full-value property-tax rate per $10,000
>        11. PTRATIO  pupil-teacher ratio by town
>        12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
>        13. LSTAT    % lower status of the population
>        14. PRICE     Median value of owner-occupied homes in $1000's
>    
>:Missing Attribute Values: None
>
>:Creator: Harrison, D. and Rubinfeld, D.L.


### Descriptive Statistics:

`us DateFrame.describe()` to show the mix, max, mean, std dev, count, and quartiles for every column.

## Visualizing the Features

### GOAL: Use Seaborn's [ `.displot()` ](https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot) to create a bar chart and superimpose the Kernel Density Estimate for
	1. PRICE
	2. RM
	3. DIS
	4. RAD
Adding titles in #seaborn : `instance.fig.suptitle("TITLE")`	
Some seaborn plots return a matplotlib Axes object, these are Axes-Level. Others are Figure-Level and return a seaborn object such as a `FacetGrid`
 #matplotlib labeling:
 `Axes.set_xlabel(label)`
 `Axes.set_ylabel(label)`
## Run  a pair plot

>- What would you expect the relationship to be between pollution (NOX) and the distance to employment (DIS)?

I would expect pollution to increase as distance to employment decreases. They would be inversely proportional.

>- What kind of relationship do you expect between the number of rooms (RM) and the home value (PRICE)?

I would expect to home value to increase as the number of rooms increases.

>- What about the amount of poverty in an area (LSTAT) and home prices?

I would expect area poverty to decrease as home prices increase.

A #pairplot allows you to visual relationships between columns.
[More on Seaborn pairplots](https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot)

## Joint Plots
#jointplot [documentation](https://seaborn.pydata.org/generated/seaborn.jointplot.html)

Using #train_test_split [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
`from sklearn.model_selection import train_test_split`

1. Create subsets
		

