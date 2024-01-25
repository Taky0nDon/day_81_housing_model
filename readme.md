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
		

```python
from sklearn.model_selection import train_test_split

X, y = data[[col for col in data.columns if col != "PRICE"]], data["PRICE"]
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=10)

price_regression = LinearRegression()
price_regression.fit(X_train, y_train)
intercept = price_regression.intercept_
slope = price_regression.coef_
print(f"intercept: {intercept}\nslope:{sorted([s for s in slope])}")
```
^ coefficients
OUT:
>intercept: 36.53305138282431
slope:[-16.271988951469734, -1.4830135966050273, -0.8203056992885642, -0.581626431182139, -0.12818065642264795, -0.012082071043592574, -0.00757627601533797, 0.011418989022213357, 0.01629221534560711, 0.06319817864608888, 0.30398820612116106, 1.9745145165622597, 3.1084562454033]
>price_regression.score(X_train, y_train)
>X_train

```
price_regression.score(X_train, y_train)
X_train
```
^^ r-squared


## Analyze the Estimated Values and Regression Residuals

Residuals are the difference between our model's prediction and the true value from `y_train`.

```python
Predicted_values = regression.predict(X_train)
residuals = (Y_train - predicted_values`)
```

### challenge one: Actual vs predicated prices:
* Actual prices on x-axis, predicted on y
* The distance of the data points from the regression line are the residuals

### Residuals vs Predicted Values
* Predicted price on x-axis
* residuals on y-axis
### Why are residuals important?
We can determine flaws in our model by analyzing the errors. If there is a pattern, that means we have a systematic error, ie: a flaw in our model. Ideally any errors would be 100% explained by chance. 

We are particularly interested in the #skew and #mean of the #residuals
A perfect bell curve has a mean distance from the mean of zero and a skew of zero, meaning the graph is completely symmetrical. 
<img src=https://i.imgur.com/7QBqDtO.png height=400>

#seaborn #format #title
```python
y_hat = price_regression.predict(X_train)
y_i = y_train
actual_vs_predicted_prices = sns.scatterplot(x=y_hat,
                                             y= y_i,
                                            )
actual_vs_predicted_prices.set_title("Actual v Predicted Prices")
actual_vs_predicted_prices.set_xlabel("Predicted Price (Y-hat)")
actual_vs_predicted_prices.set_ylabel("Actual price (yi)")
```
![[Pasted image 20240124182816.png]]
k
Now do predicted price on the x-axis and residual(actual - minus predicted) on the y-axis

#kde super impose kde over histogram representation of a Series:

```python
res_plot = sns.displot(x=residuals, kde=True)
```

![[Pasted image 20240124190502.png]]

## Data transformations

At this point we must either consider a new model entirely, our transforming our data (**actual**) to make it better fit with our linear model

Is `data["PRICE"]` a good candidate for log transformation?
```python
price_plot = sns.displot(x=data["PRICE"],
                         kde=True,
                        )
price_skew = data.PRICE.skew()
print(f"Our price data has a skew of {price_skew}")
```
![[Pasted image 20240124191144.png]]

Use `NumPy.log()` to create a series with the logarithmic prices
Compare the skews. Which is closer to zero?
```
log_price = np.log(data.PRICE)
print(f"The log data has a skew of {log_price.skew()}")
log_price_plot = sns.displot(x=log_price,
                             kde=True,
                            )
```
![[Pasted image 20240124191355.png]]
The log transformation has a skew much closer to 0.

#### How does the #log #transformation work?
* Every datum is replaced by it's `ln` (natural log)
* Large value are more affected that smaller ones. They are 'compressed'
<img src=https://i.imgur.com/TH8sK1Q.png height=200>

If we use log prices, our model becomes: $$ \log (PR \hat ICE) = \theta _0 + \theta _1 RM + \theta _2 NOX + \theta_3 DIS + \theta _4 CHAS + ... + \theta _{13} LSTAT $$

