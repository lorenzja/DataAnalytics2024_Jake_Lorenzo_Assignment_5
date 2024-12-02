# DataAnalytics2024_Jake_Lorenzo_Assignment_5


Part 1:
A. Describe the type of patterns or trends you might look for and how you plan to explore 
and model them. (Min. 3 sentences)

I would look for patterns related to sales price and see which factors most signifacantly influence it. To do this I would perform a multivariate linear model to see which factor/feature most signifacantly infleunces Sale Price. I could also look to see if there is any fluctuation in sales price related to time. That could potentially give insight into the NYC housing market's seasonal trends or strength of the market as a whole. I could plan to explore this my filtering the dataset by time and see if that gives me any significant data. 

B. Perform exploratory data analysis (variable distributions, etc.) and describe what you 
did including plots and other descriptions. Identify the outlier values in the data for Sale 
Price and generate suitable plots to demonstrate the outliers relative to the other data 
points. (Min. 5 sentences) 

I began by doing preliminery data preprocessing including filtering the dataset down to one borough (Manhattan). I then removed zero values, NA values, and -0 values. These values will create errors (negative infinity) later in the code when converting data columns to log base 10 form for input into the linear model. I then removed several columns from the dataset that I felt would not be relevant, as well as, transformed several feature columns to either factor form or numerical form depending on if the feature was categorical or not. I then split the data into X and Y for plotting. I created some preliminery plots for X and Y as well as combining X and Y to create a pairs.panels plot and a ggplot depicting Sales Price in comparison to GROSS SQUARE FEET. From these plots I was able to identify outliers in the Sales Price column which resulted in removing rows from the dataset that contained a Sales Price values less than 1,000 or greater than 1 Billion.

C. Conduct Multivariate Regression on the 1 borough dataset to predict the Sale Price 
using other variables that may have a meaningful connection to price. After you identify a 
well-performing model test it on 2 subsets of 1 borough dataset (based on meaningful 
criteria of your choice, such as building class or sq footage value) and compare the results 
you obtained. You may have to try multiple models and drop variables with very low 
significance. Explain the results. (Min. 5 sentences) 

D. Pick more than one supervised learning model (these need not be restricted to the 
models you’ve learned so far), e.g., Naïve Bayes, k-NN, Random Forest, SVM to explore a 
classification problem using the data. You may choose which categorical variable (e.g. 
neighborhood, building class) to use as class label. Evaluate the results (contingency 
tables & metrics). Describe any cleaning you had to do and why. (Min. 5 sentences)

Part 2:

A. Apply the best performing regression model(s) from 1.c to predict Sale Price based on 
the variables you chose. Plot the predictions and residuals. Explain how well (or not) the 
models generalize to the whole dataset and speculate as to the reason. (Min. 3-4 sentences)

B. Apply the classification model(s) from 1.d to predict the categorical variable of your 
choice. Evaluate the results (contingency tables & metrics). Explain how well (or not) the 
models generalize to the whole dataset and speculate as to the reason. (Min. 3-4 sentences)

C.  Discuss any observations you had about the datasets/ variables, other data in the 
dataset and/or your confidence in the result. (Min 1-2 sentences)

Part 3 (Grad Level):
Draw conclusions from this study – about the model type and 
suitability/ deficiencies. Describe what worked and why/ why not. (Min. 6-7 sentences)

