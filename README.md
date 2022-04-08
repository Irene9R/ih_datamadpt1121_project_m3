<p align="left"><img src="https://cdn-images-1.medium.com/max/184/1*2GDcaeYIx_bQAZLxWM4PsQ@2x.png"></p>

# __IPM Diamonds__ #
Irene Rengifo - Project Module 2 - Data Analytics Part-Time Bootcamp - Nov 2021 - Ironhack Madrid
## **Goal** ##
To create a dashboard that is user friendly, that the customer understands as much as possible about diamonds. 

## **Overview** ##
### **Data Exploration and Preparation.** ### 

The goal of this challenge is to perform an exploratory analysis in order to gain initial insight on our diamonds database and prepare the data model that better fits your visualizations. 

#### **Data Collection** ####
I used Jupyter Notebook and SQLite to connect and extract the information from diamonds.db database.  Using pandas and numpy libraries, I then exported the diamonds.csv to start working with:

#### **Exploratory Analysis** ####
Descriptive statistics with Tableau
First insights of the data based on descriptive statistics
Graphical representation of descriptive statistics and relations


![Image](https://ae01.alicdn.com/kf/H19dd6da48ab04e43ba543e893827f6b4I.jpg)

### **Dashboard** ### 

Identification of needs to develop the dashboard and development of the dashboard

Aim: To create a dashboard that is user friendly, that the customer understands as much as possible about diamonds. My dashboard is made so that you can see everything, however, within the options you have in terms of price, you can see the diamond options that exist; and compare between the diamonds (within the parameters) having the highest visibility to differentiate the options (cut, color, clarity).

The parameters I worked with were price and caratage group.

In order to achieve this, I needed to create a dynamic dashboard. Used parameters and calculated fields to get the dashboard to present as much information as possible in a summarized way.


## **Data** ##
1. diamonds.db 

## **Project Main Stack**

- [SLQlite](https://www.sqlite.org/index.html) - used during the starting process of working on the project. 
- [Tableau Public](https://public.tableau.com/en-us/s/) 
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)




## **How to access IPM Diamonds Visualizations**  

[Dashboard](https://public.tableau.com/app/profile/irene1690/viz/DiamondDashboard-IPMDiamonds/Dashboard1?publish=yes)

[Tableau Public - Visualization Story](https://public.tableau.com/app/profile/irene1690/viz/DiamondStatistics-IPMDiamonds/Story1?publish=yes)



## **Conclusions* #

There are some outlier values in the dataset that affects the **mean price** of the diamonds, regardless of the group **category**. 

While I did the analysis with the main dataset by category, I found some odd results that would suggest that diamonds with lower level color or with lower quality of **cuts** had a **higher average price** than others with higher quality. I concluded that this was due to comparing diamonds from different carats between each other.  
For this reason, I decided that to obtain the most objective result, I should compare all characteristics within diamonds of the same **group of carats**, by doing this I found that my new results had more logic, and that indeed while comparing between diamonds of the same range of carats, higher level **color**, **clarity** and **cuts** did have a **higher average price** as compared to lower quality within the same carat group. 

Insights: 
1.	Most diamonds in this dataset have **“Ideal”** cut and **“G”** color.
2.	The cities with the highest amount of diamonds and **total sales** are Surat, Kimberly and Antwerp. 
3.	In conclusion, I found that diamonds with higher quality in all the parameters have higher average prices when compared to others within the same carat group. 

For me, the main conclusion that I've obtained by working with this diamonds database is that the quality of a diamond is explained by different features that are objectively measured by professionals. Some of them are more important than others. This is the thing with carats: the bigger, the more expensive. But this only represents the size of the diamond, not the quality in terms of the other features.


## **and remember: if you don't have a diamond, YOU CAN'T SIT WITH US!!!.** ##
![Image](https://media.giphy.com/media/xT9KVuimKtly3zoJ0Y/giphy.gif)

Regression-based machine learning approaches for diamond price prediction!

Photo by Sina Katirachi on Unsplash
Inthis blog-post, we will go through the whole process of creating a machine learning model on a diamond price prediction dataset. Then we will also compare the results using various regression metrics.
You can also download the dataset from this repository. It contains the information about dimension, color, clarity, weight, and cut of diamonds vs price.
1. Diamonds

Fig. 1. Diamonds
Diamond forms under high temperature and pressure conditions that exist only about 100 miles beneath the earth’s surface. Diamond’s carbon atoms are bonded in essentially the same way in all directions. Another mineral, graphite, also contains only carbon, but its formation process and crystal structure are very different. Diamonds have been used as decorative items since ancient times; some of the earliest references can be traced back to 25,000–30,000 B.C.
Facts
Mineral: Diamond
Chemistry: C
Color: Colorless
Refractive Index: 2.42
Birefringence: None
Specific Gravity: 3.52 (+/-0.01)
Mohs Hardness: 10
Currently, gem production totals nearly 30 million carats (6.0 tonnes; 6.6 short tons) of cut and polished stones annually, and over 100 million carats (20 tonnes; 22 short tons) of mined diamonds are sold for industrial use each year, as are about 100 tonnes (110 short tons) of synthesized diamond. Diamonds are such a highly traded commodity that multiple organizations have been created for grading and certifying them based on the “four Cs”, which are color, cut, clarity, and carat. You can read about “four Cs” from this link.
2. Dataset

Photo by Mika Baumeister on Unsplash
As it is well said, “A good craftsman always knows his tool”, in the same way, a good ML developer should always know his/her dataset properly before actually starting to code. It’s great to follow this habit because a machine learning model is an image of the dataset used for training it. So, it’s likely that any trend or pattern followed datasets would reflect in the machine learning model. According to my experience, 80% of work is already done when you have a good preprocessed dataset.
So, let’s explore the unexplored!
Note: I would suggest to use google colab, as it has made my life easier by handling all the dependencies and libraries, moreover it also provides utilities like GPU and TPU for computation and it’s also easy to directly link the google drive with your code.
First of all, let’s create a data frame and visualize the dataset.
# import necessities
import pandas as pd
import numpy as np
from sklearn import preprocessing
df = pd.read_csv("/content/..../diamonds.csv")
df

Fig. 2. Dataset
Now, let’s see all the necessary information.
df.info()
Output:
<class 'pandas.core.frame.DataFrame'> 
RangeIndex: 53940 entries, 0 to 53939 
Data columns (total 10 columns): 
carat      53940 non-null float64 cut        53940 non-null object color      53940 non-null object clarity    53940 non-null object depth      53940 non-null float64 table      53940 non-null float64 price      53940 non-null int64 
x          53940 non-null float64 
y          53940 non-null float64 
z          53940 non-null float64 
dtypes: float64(6), int64(1), object(3) 
memory usage: 4.1+ MB
Hmmm, looks fine!
Now, let’s get a look at entries whose datatype is an object.
print("Cut: ",set(df["cut"]))
print("Color: ",set(df["color"]))
print("Clarity: ",set(df["clarity"]))
Output:
Cut: {'Fair', 'Good', 'Ideal', 'Premium', 'Very Good'}
Color: {'D', 'E', 'F', 'G', 'H', 'I', 'J'}
Clarity: {'I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'}
Hmmm, we have a limited number of categorical values, we need to process this somehow as our model won’t understand non-numerical data!
3. Pre-processing

Photo by Markus Spiske on Unsplash
After understanding the data, now it’s time to process the data in such a way that a machine could understand it as unlike humans, a machine could just understand numbers, as ML models are nothing but complex probability equations.
3.1. Encoding
As I mentioned above that there is a need to convert non-numerical data into a numerical one and three of our features are non-numerical, namely “cut”, “color” and “clarity”. We can easily apply one-hot encoding to this problem but that would simply increase the number of features and so the computation power required. Instead, let’s try to derive a relation and map those categories.
Note: To formulate relationships with the price of diamond we must have a standard reference in which we can compare values, so here I have compared the prices per unit carat.
df['price/wt']=df['price']/df['carat']
print(df.groupby('cut')['price/wt'].mean().sort_values())
print(df.groupby('color')['price/wt'].mean().sort_values())
print(df.groupby('clarity')['price/wt'].mean().sort_values())
df = df.drop(['price/wt','table'], axis=1)
Output:
cut 
Fair         3767.255681 
Good         3860.027680 
Ideal        3919.699825 
Very Good    4014.128366 
Premium      4222.905374 
Name: price/wt, dtype: float64 
color 
E    3804.611475 
J    3825.649192 
D    3952.564280 
I    3996.402051 
H    4008.026941 
F    4134.730684 
G    4163.411524 
Name: price/wt, dtype: float64 
clarity 
I1      2796.296437 
SI1     3849.078018 
VVS1    3851.410558 
SI2     4010.853865 
VS2     4080.526787 
VS1     4155.816808 
VVS2    4204.166013 
IF      4259.931736 
Name: price/wt, dtype: float64
Nice! We can easily notice a trend in values in all three features. This gives us the flexibility to map categorical values with numbers.
df['cut']=df['cut'].map({'Ideal':1,'Good':2,'Very Good':3,'Fair':4,'Premium':5})
df['color']=df['color'].map({'E':1,'D':2,'F':3,'G':4,'H':5,'I':6,'J':7})
df['clarity']=df['clarity'].map({'VVS1':1,'IF':2,'VVS2':3,'VS1':4,'I1':5,'VS2':6,'SI1':7,'SI2':8})
3.2. Feature selection using the correlation matrix

Photo by Inês Ferreira on Unsplash
One last thing, let’s view the correlation matrix for our data. A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between the two variables. Higher the value more likely the data correlates! For, this problem as we want to predict the price of a diamond, we will focus on the correlation between price vs all other columns.
df.corr()

Fig. 3. Correlation Matrix
Bravo!!!
Now, from this information, we can apply feature selection on the data. It’s clear that the following features have a low score, it means they are least correlated with the price of a diamond:
cut
color
clarity
depth
table

Fig. 4. Diamond Glossary
Although, these features have a low correlation score, yet we can’t remove them. For a gemologist features like “cut”, “color” and “clarity” play a crucial role in determining the price of a diamond.
On the other hand, if we study the correlation matrix, it’s clear that the price and carat i.e. weight of a diamond has a very good correlation score. So, we can insert new features like cut per weight score and drop the features like “cut”, “color” and “clarity”. Similarly, we can interpret “table” by multiplying it with “y” and then drop “depth” because of a low score.
df['cut/wt']=df['cut']/df['carat']
df['color/wt']=df['color']/df['carat']
df['clarity/wt']=df['clarity']/df['carat']
df = df.drop(['cut','color','clarity','table','depth'], axis=1)

Fig. 5. Final Correlation Matrix
Fabulous!!!
4. Train-test split
Now, after pre-processing the dataset, we need to split it into training and testing set to avoid overfitting or underfitting problems as it enables us to evaluate the prediction by various regression matrices.
X=df.drop(['price'],axis=1)
Y=df['price']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
Note: Using random_state=some_number, we guarantee that the split will be always the same. Moreover, random_state=42 is commonly used because of this reason 😉.
6. Regression Metrics

Photo by Helloquence on Unsplash
Step 6 before step 5? No, I am not joking, but let’s be clear how we would evaluate and compare different models. I would be using the following regression metrics:
1. Mean absolute error: Mean Absolute Error refers to the mean of the absolute values of each prediction error on all instances of the test data-set. The prediction error is the difference between the actual value and the predicted value for that instance.

Fig. 6. Mean absolute error
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
2. R-squared score: R-squared is a statistical measure that states how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

Fig. 7. R-squared score
Rsquare=regressor.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
3. Root mean squared error: It is the square root of the sum of the square of the difference between the predicted and actual target variables, divided by the number of data points.

Fig. 8. Root mean squared error
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
5. Let’s make the machine learn!

Photo by Hitesh Choudhary on Unsplash
After collecting and processing all the ingredients now it’s time to try different recipes and see which one tastes the best.
5.1. Linear Regression
Linear regression is used for finding a linear relationship between the target and one or more predictors.

Fig. 9. Linear Regression Equation
from sklearn import linear_model
reg_all=linear_model.LinearRegression()
reg_all.fit(X_train,Y_train)
y_pred=reg_all.predict(X_test)
Rsquare=reg_all.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Variable']
coeff_df["Coeff"] = pd.Series(reg_all.coef_)
coeff_df.sort_values(by='Coeff', ascending=True)
print(coeff_df)
print("Intercept: %f" %(reg_all.intercept_))
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
Output:
Rsquare: 0.861508
      Variable         Coeff 
0       carat  10675.558686 
1           x  -1192.078847 
2           y    957.349861 
3           z   -615.526175 
4   width_top    -15.053996 
5      cut/wt     30.065902 
6    color/wt    -48.086153 
7  clarity/wt    -64.338733 
Intercept: 4625.114040 
mae: 895.345484 
rmse: 1469.665116
5.2. Polynomial Regression
Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y and has been used to describe nonlinear phenomena.

Fig. 10. Polynomial Regression Equation
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
reg_all = linear_model.LinearRegression()
reg_all.fit(X_train,Y_train)
y_pred=reg_all.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
Rsquare=reg_all.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
Output:
mae: 549.500225 
Rsquare: 0.933456 
rmse: 1018.734087
And for degree=3 we got the following output:
mae: 912.372488 
Rsquare: -188.653181 
rmse: 54385.881154
5.3. Decision tree Regression

Photo by Luke Richardson on Unsplash
A decision tree is a supervised machine learning model used to predict a target by learning decision rules from features. As the name suggests, we can think of this model as breaking down our data by making a decision based on asking a series of questions.

Fig. 11. Decision tree Regression
A decision tree is constructed by recursive partitioning — starting from the root node (known as the first parent), each node can be split into left and right child nodes. These nodes can then be further split and become parent nodes of their resulting children nodes.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
Rsquare=regressor.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
Output:
mae: 373.421487 
Rsquare: 0.961235 
rmse: 777.550255
5.4. Support Vector Regression
Support Vector Machine can also be used as a regression method, maintaining all the main features that characterize the algorithm (maximal margin). The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. First of all, because the output is a real number it becomes very difficult to predict the information at hand, which has infinite possibilities. In the case of regression, a margin of tolerance (epsilon) is set in approximation to the SVM which would have already requested from the problem. But besides this fact, there is also a more complicated reason, the algorithm is more complicated therefore to be taken into consideration. However, the main idea is always the same: to minimize error, individualizing the hyperplane which maximizes the margin, keeping in mind that part of the error is tolerated.

Fig. 12. Support Vector Regression
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
Rsquare=regressor.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
Note: The code might take a little more time to run.
Output:
mae: 2672.065590 
Rsquare: -0.097922 
rmse: 4138.012816
5.5. Random Forest

Photo by Sebastian Unrau on Unsplash
Random forest is a Supervised Learning algorithm which uses ensemble learning method for classification and regression.
Random forest is a bagging technique and not a boosting technique. The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.
It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
A random forest is a meta-estimator (i.e. it combines the result of multiple predictions) which aggregates many decision trees.

Fig. 13. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 6)
rf.fit(X_train,Y_train)
y_pred = rf.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
Rsquare=rf.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
Note: The following code will take a lot of time to run as it is computationally heavy.
Output:
mae: 662.102892 
Rsquare: 0.119454 
rmse: 1423.369221
5.6. Logistic Regression

Fig. 14. Logistic Regression
Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
Logistic Regression is one of the most popular ways to fit models for categorical data, especially for binary response data in Data Modeling. Hence, it won’t work well in our case.
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 0)
logreg.fit(X_train,Y_train) 
y_pred=logreg.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)
print("mae: %f" %(mae))
Rsquare=logreg.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
Output:
mae: 1294.865468 
Rsquare: 0.011185 
rmse: 2107.933862
5.6. Neural Network

Photo by Christopher Burns on Unsplash

Fig. 15. Neural Network Architecture
As the amount of data increases, the performance of traditional learning algorithms, like SVM and logistic regression, does not improve by a whole lot. In fact, it tends to plateau after a certain point. In the case of neural networks, the performance of the model increases with an increase in the data you feed to the model.
If the network is used for regression then the training loss function is normally nothing at all, usually, the mean square error (MSE) or root mean square error (RMSE) is used as a metric within the loss function.
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential,model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(8,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','mean_squared_error'])history = model.fit(X_train,Y_train,batch_size=64,epochs=100,verbose=2)
test=model.evaluate(X_test, Y_test, verbose=1)
Note: The model was trained for 100 epochs.
Output:
Using TensorFlow backend.
Model: "sequential_1" _________________________________________________________________ Layer (type)                 Output Shape              Param #    ================================================================= dense_1 (Dense)              (None, 256)               2304       _________________________________________________________________ dense_2 (Dense)              (None, 256)               65792      _________________________________________________________________ dense_3 (Dense)              (None, 256)               65792      _________________________________________________________________ dense_4 (Dense)              (None, 256)               65792      _________________________________________________________________ dense_5 (Dense)              (None, 1)                 257        ================================================================= Total params: 199,937 
Trainable params: 199,937 
Non-trainable params: 0
- 2s - loss: 1525.3224 - mean_absolute_error: 1525.3224 - mean_squared_error: 8729575.7525 Epoch 2/100  
- 2s - loss: 684.7146 - mean_absolute_error: 684.7146 - mean_squared_error: 1974301.8159 Epoch 3/100  
- 2s - loss: 672.9809 - mean_absolute_error: 672.9809 - mean_squared_error: 1829558.6347 Epoch 4/100
.
.
.
- 2s - loss: 395.8742 - mean_absolute_error: 395.8742 - mean_squared_error: 793110.1403 Epoch 99/100  
- 2s - loss: 403.3112 - mean_absolute_error: 403.3112 - mean_squared_error: 796024.6950 Epoch 100/100  
- 2s - loss: 406.1216 - mean_absolute_error: 406.1216 - mean_squared_error: 809139.8724 16182/16182 [==============================] - 0s 21us/step
Note: We can take the square root of mean squared error (mse) to get the root mean squared error (rmse), which make rmse=899.522024 .
5.7. Regularization
One of the major aspects of training your machine learning model is avoiding overfitting. The model will have a low accuracy if it is overfitting. This happens because your model is trying too hard to capture the noise in your training dataset. By noise, we mean the data points that don’t represent the true properties of your data, but random chance. Learning such data points, makes your model more flexible, at the risk of overfitting. Regularization is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, to avoid the risk of overfitting.
Ridge and Lasso methods are used for regularization. Generally, regularization is combined with regression techniques to avoid any overfitting.

Fig. 16. The cost function for regularization
regressor1 = Ridge()
regressor2 = Lasso()

Fig. 16. The output of Ridge and Lasso
7. Summary



Decision tree Regression proved to be the best module I tried for a diamond’s price prediction which was shortly followed by Neural Networks.

Thanks for reading my blog, I am happy to share my knowledge 😃.










