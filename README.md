<p align="left"><img src="https://cdn-images-1.medium.com/max/184/1*2GDcaeYIx_bQAZLxWM4PsQ@2x.png"></p>

# __IPM Diamond Prediction's at your service!__ #
Irene Rengifo - Project Module 3 - Data Analytics Part-Time Bootcamp - Nov 2021 - Ironhack Madrid

![Image](https://media.baamboozle.com/uploads/images/161225/1617025270_318369_gif-url.gif)
	

## **Goal** ##
In this Kaggle competition the aim is to predict the price of diamonds based on their characteristics. You should get the minimum RMSE to win the competition. 

## **Overview** ##

We will go through the whole process of creating a machine learning model on a diamond price prediction dataset. Then we will also compare the results using various regression metrics.
You can also download the data from kaggle: https://www.kaggle.com/competitions/dataptmad1121/datain  


## **Data** ##
1. **The training set:**  
"diamonds_train.db" -  I used Jupyter Notebook and SQLite to connect and extract the information from diamonds.db database.  Using pandas and numpy libraries, I then exported the diamonds.csv to work with it.
2. **The test dataset:**  
"diamonds_test.csv"
3. **Submission sample:**  
"sample_submission.csv" - a sample submission file in the correct format.  
  
  
#### Feature Description ####
**price*:  price in USD  
**carat*:  weight of the diamond  
**cut*:  quality of the cut (Fair, Good, Very Good, Premium, Ideal)  
**color*:  diamond colour, from J (worst) to D (best)  
**clarity*:  a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))  
**x*:  length in mm  
**y*: width in mm  
**z*: depth in mm  
**depth*: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)  
**table*: width of top of diamond relative to widest point (43--95)  
**city*: city where the diamonds is reported to be sold.  
**id*: only for test & sample submission files, id for prediction sample identification.  
  

## **Project Main Stack**

- [SLQlite](https://www.sqlite.org/index.html) - used during the starting process of working on the project. 
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)
- [sklearn](https://scikit-learn.org/stable/)  
- [Matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)


## Exploratory Data Analysis (EDA) ##

As it is well said,**“A good craftsman always knows his tool”**, in the same way, a good ML developer should always know his/her dataset properly before actually starting to code. It’s great to follow this habit because a machine learning model is an image of the dataset used for training it. So, it’s likely that any trend or pattern followed datasets would reflect in the machine learning model. 

### So, the first step was to explore the unexplored! ###

![Image](https://cdn.vox-cdn.com/thumbor/r0VBSdDbNeBrMq8HOxqVN_kjjqQ=/0x0:2700x1350/2420x1613/filters:focal(1134x459:1566x891):format(webp)/cdn.vox-cdn.com/uploads/chorus_image/image/58026259/jumanjicover.0.jpg)


I started by evaluating the data using .unique(), value_counts(), isnull() and others, to help me evaluate my dataset: Are there were any null or missing values in the dataset?, are there any signs of wrongly entered data?, but the dataset is very clean as we already used it for project 2. 


Reviewing the datatype of each column, is visible that there are a limited number of categorical values in the dataset that will become the first challenge, as they need to be processed since the model won’t understand non-numerical data. 

## Pre-processing ##

After understanding the data, now it’s time to process the data in such a way that a machine could understand it. As unlike humans, a machine only understand numbers, therefore ML models are complex probability equations.

#### 1. Encoding ####
As I mentioned before, there is a need to convert non-numerical data into numerical ones and three of our features are non-numerical. This features are: “cut”, “color” and “clarity”. 
I could apply one-hot encoding to this problem but that would increase the number of features. Instead, I decided to derive a relation and map those categories.  
**Note**: To formulate relationships with the price of diamond I needed to have a standard reference to compare values, so I compared the prices per unit carat.

There is a noticeable trend in values in all three features. This gives the flexibility to map categorical values with numbers.

#### 2. Feature selection using the correlation matrix ####

I used a correlation matrix and plot the heatmap in order to get the correlation coefficients between variables. Since I want to predict the price of a diamond, I focused on the correlation between price vs all other columns using .corr().

After reviewing it, it was clear that the following features had a low score, this means they are least correlated with the price of a diamond:  
cut  
color  
clarity  
depth  
table  


Although, these features have a low correlation score, I won't remove them at the moment. For a diamond specialist features like “cut”, “color” and “clarity” play a crucial role in determining the price of a diamond.
On the other hand, studying the correlation matrix, it’s clear that the price and carat i.e. weight of a diamond has a very good correlation score. So, I inserted new features, like "cut per weight score" and drop the features like “cut”, “color” and “clarity”. 

#### 3. Scaling ####
I have tried different methods of scaling like "StandardScaler()", "MinMaxScaler()", "RobustScaler()", "MaxAbsScaler()"; but ended up not using a scaling method as it didn´t decrease ed RMSE, more on the contrary, so I decided to work without using it. 


## Train-test split ##

After pre-processing the dataset, since I wanted to  to try to make as many tests as possible on my own and the test dataset didn´t have the target column to predict (in this case "price"), the only way for me to make tests before submitting, was to split my training dataset  into training and testing set to avoid overfitting or underfitting problems as it enables us to evaluate the prediction by various regression matrices.
Note: Using random_state=some_number, we guarantee that the split will be always the same. Moreover, random_state=42 is commonly used because of this reason. 

## Regression Metrics ##

This will actually be performed after we  train, test and predict with our modules, BUT since I will be using the following regression metrics to evaluate and compare the performance of the different modules, I would think is better to explain them before hand.   

1. **Mean absolute error**  
Mean Absolute Error refers to the mean of the absolute values of each prediction error on all instances of the test data-set. The prediction error is the difference between the actual value and the predicted value for that instance.

2. **R-squared score**  
R-squared is a statistical measure that states how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

3. **Root mean squared error**  
 It is the square root of the sum of the square of the difference between the predicted and actual target variables, divided by the number of data points.

## Now, making the machine learn! ##

![Image](https://c.tenor.com/aOEhTr-O9XAAAAAC/commando-arnold-schwarzenegger.gif)
	

After collecting and processing all the ingredients now it’s time to try different recipes and see which one tastes the best.
I worked with several models like:
1. **Linear Regression:**  
Linear regression is used for finding a linear relationship between the target and one or more predictors. 
2. **ElasticNet**   
ElasticNet is a linear regression model trained with both l1 and l2 -norm regularization of the coefficients. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. We control the convex combination of l1 and l2 using the l1_ratio parameter."
3. **Decision tree Regression:**  
A decision tree is a supervised machine learning model used to predict a target by learning decision rules from features. As the name suggests, we can think of this model as breaking down our data by making a decision based on asking a series of questions.  
4. **Stochastic Gradient Descent (SGD)**  
SGD is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.
5. **RandomForestRegressor:**   
The sklearn.ensemble module includes two averaging algorithms based on randomized decision trees: the RandomForest algorithm and the Extra-Trees method. Both algorithms are perturb-and-combine techniques [B1998] specifically designed for trees. This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.  
6. **GradientBoostingRegressor:**  
Gradient Tree Boosting or Gradient Boosted Decision Trees (GBDT) is a generalization of boosting to arbitrary differentiable loss functions. GBDT is an accurate and effective off-the-shelf procedure that can be used for both regression and classification problems in a variety of areas including Web search ranking and ecology.
The module sklearn.ensemble provides methods for both classification and regression via gradient boosted decision trees.
7. **Regularization**  
One of the major aspects of training your machine learning model is avoiding overfitting. The model will have a low accuracy if it is overfitting. This happens because your model is trying too hard to capture the noise in your training dataset. By noise, we mean the data points that don’t represent the true properties of your data, but random chance. Learning such data points, makes your model more flexible, at the risk of overfitting. Regularization is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, to avoid the risk of overfitting.
Ridge and Lasso methods are used for regularization. Generally, regularization is combined with regression techniques to avoid any overfitting.

## Hyper-Parameters Optimization or Tunning ##

**RandomizedSearchCV**  

While using a grid of parameter settings is currently the most widely used method for parameter optimization, other search methods have more favourable properties. RandomizedSearchCV implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values. This has two main benefits over an exhaustive search:

- A budget can be chosen independent of the number of parameters and possible values.  
- Adding parameters that do not influence the performance does not decrease efficiency.

## My Best Prediction ## 

The best result I got was using the RandomForestRegressor() since it obtained the lowest RMSE. 

Even if not the best prediction in the class, I'm proud since I learned something new and I'm  

## The best part is not always to win, but to learn! ## 

![Image](https://peacewords.us/wp-content/uploads/2021/03/NelsonMandelaquotes-.jpg)







































































