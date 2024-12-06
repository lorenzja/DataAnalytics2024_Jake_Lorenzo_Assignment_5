# DataAnalytics2024_Jake_Lorenzo_Assignment_5


Part 1:
A. Describe the type of patterns or trends you might look for and how you plan to explore 
and model them. (Min. 3 sentences)

I would look for patterns related to sales price and see which factors most signifacantly influence it. To do this I would perform a multivariate linear model to see which factor/feature most signifacantly infleunces Sale Price. I could also look to see if there is any fluctuation in sales price related to time. That could potentially give insight into the NYC housing market's seasonal trends or strength of the market as a whole. I could plan to explore this by filtering the dataset by time and see if that gives me any significant data. 

B. Perform exploratory data analysis (variable distributions, etc.) and describe what you 
did including plots and other descriptions. Identify the outlier values in the data for Sale 
Price and generate suitable plots to demonstrate the outliers relative to the other data 
points. (Min. 5 sentences) 

I began by doing preliminery data preprocessing including filtering the dataset down to one borough (Manhattan). I then removed zero values, NA values, and -0 values. These values will create errors (negative infinity) later in the code when converting data columns to log base 10 form for input into the linear model. I then removed several columns from the dataset that I felt would not be relevant, as well as, transformed several feature columns to either factor form or numerical form depending on if the feature was categorical or not. I then split the data into X and Y for plotting. I created some preliminery plots for X and Y as well as combining X and Y to create a pairs.panels plot and a ggplot depicting Sales Price in comparison to GROSS SQUARE FEET. From these plots I was able to identify outliers in the Sales Price column which resulted in removing rows from the dataset that contained a Sales Price values less than 50,000 or greater than 200 Million. I also removed rows that contained values in the GROSS SQUARE FEET column larger than 400,000 square feet.

C. Conduct Multivariate Regression on the 1 borough dataset to predict the Sale Price 
using other variables that may have a meaningful connection to price. After you identify a 
well-performing model test it on 2 subsets of 1 borough dataset (based on meaningful 
criteria of your choice, such as building class or sq footage value) and compare the results  
you obtained. You may have to try multiple models and drop variables with very low 
significance. Explain the results. (Min. 5 sentences) 

I ran into trouble using 'factor' variables when using the linear model. I had a lot of errors regarding 'differing levels' when plotting predicted vs expected Sale Price. I was able to remedy this situation by training the linear model on the entire dataset and then testing the accuracy on the test set. this led me to get a r-squared of 1 and an adjusted r-squared of 1 with the plot as a perfect straight line. This led me to remove all of the 'factor' variables from the model and only use "GROSS SQUARE FEET, LAND SQUARE FEET, YEAR BUILT, and TOTAL UNTIS. After running the initial model I conducted further outlier removal by removing very large or small values from the SALE PRICE ( > 100,000,000 ) and GROSS SQUARE FEET ( > 50,000) columns. This led me to getting a Multiple R-squared:  0.4052,	Adjusted R-squared:  0.4046. This could be improved with further refinement of the model and further outlier processing and removals. I could also look at the plots of the data to look for any patterns or `errors` in the data.

D. Pick more than one supervised learning model (these need not be restricted to the 
models you’ve learned so far), e.g., Naïve Bayes, k-NN, Random Forest, SVM to explore a 
classification problem using the data. You may choose which categorical variable (e.g. 
neighborhood, building class) to use as class label. Evaluate the results (contingency 
tables & metrics). Describe any cleaning you had to do and why. (Min. 5 sentences)

I initially ran a SVM classification model with a linear kernal to predict LOT based on the data from the other columns. I initially tried to get the recall, precision, and f1 scores of the model but it returned those scores for each unique level of LOT(of which there were thousands). I then computed the accuracy score of the model which was  "Accuracy: 0.581899298916507." This was a higher accuracy score than the KNN model however the accuracy score still isn't very high which means that outliers remain in the dataset. Additionally running a tuned SVM model would hopefully increase the model's accuracy score however my computer does not have the processing power to do that in any normal amount of time.

The second supervised learning model I used was a KNN model. I decided to use the same as above and try to predict LOT based on the information from the other columns. I took the square root of the number of observations in the dataset and used 67 as my k value. I then ran the model and got "Accuracy: 0.145724907063197". I than ran a model to use different values of k to iterate through the model and see which one gave the most accurate score. k = 67 was the most accurate k from the ones I tested. This is a very low accuracy score and would indicate same as the model above that there are still outliers remaining in the dataset.


Part 2:

A. Apply the best performing regression model(s) from 1.c to predict Sale Price based on 
the variables you chose. Plot the predictions and residuals. Explain how well (or not) the 
models generalize to the whole dataset and speculate as to the reason. (Min. 3-4 sentences)

I chose the multivariate linear model that I used above with "GROSS SQUARE FEET, LAND SQUARE FEET, YEAR BUILT, and TOTAL UNTIS" as the columns used to predict Sale Price and applyed it to the entire NYC dataset with all five boroughs. I did similar outlier removal of SALE PRICE > 75,000,000 and GROSS SQUARE FEET > 50,000 as well as feature plotting.  Multiple R-squared:  0.3934,	Adjusted R-squared:  0.3934. Looking at the R-squared values as well as plotting the linear model and its predicted vs expected values there are definetely still some outlier values that are leveraging the model. Further data munging would be required to find those values and remove them to create a more accurate model. Overall the model does not generalize the data as a whole very well and I think this is due to several factors: diversity of real estate in NYC (square footage is not always synonmous with price i.e. penthouse apartment) and there could be the presence of multicollinearity in the data.

B. Apply the classification model(s) from 1.d to predict the categorical variable of your 
choice. Evaluate the results (contingency tables & metrics). Explain how well (or not) the 
models generalize to the whole dataset and speculate as to the reason. (Min. 3-4 sentences)

I tried to apply the SVM model that I did above to the entire dataset but got the following error due to the size of the dataset. > svm.mod0 <- svm(`LOT` ~ ., data = train, kernel = 'linear') Error: cannot allocate vector of size 13.6 Gb. Due to this I decided to use the much less accurate KNN model from above and apply it to the entire NYC dataset. I decided to take the square root of the training set and made the inital k = 369 which gave a very low initial Accuracy: 0.0158472928252192. I then iterated many times through a multitude of different values of k and decided the most accurate was k = 3 with an "Accuracy: 0.0581863464228447." This model does not generalize the whole dataset very well and I would not recommend someone deploy it to try and use the various factor columns to predict Sale Price. I think this is due to high dimensionality and noise in the data. Also LOT is not a `primary` type variable and due to this the KNN model would probably struggle. I would definitely recommend the SVM model. 

C.  Discuss any observations you had about the datasets/ variables, other data in the 
dataset and/or your confidence in the result. (Min 1-2 sentences)

The dataset had a lot of noise and dimmensionality issues. I think that the data modeling I did was good to get a picture of the dataset as a whole and the interactions between different factors, however I do think that a deep dive into the data munging aspect of cleaning out the data would be the best way to optimize the various models so that they can give meaningful results with a fairly high accuracy. There is definitely a correlation between the square foot aspect and sale price although due to outliers (i am assuming from high price low square-feet penthouse apartments). Definitely a medium to high correlation betwwen LOT & BLOCK & ZIP CODE and Sale Price. Also the high correlation between LOT & BLOCK & ZIP CODE could lead to a multicollinearity effect that could affect the accuracy of the models. I would also assume a correlation between year built and sale price  with newer buildings having a positive effect on sale price. Inversely however I would think that certain historic older buildings would fetch a higher sell price as well. 

Part 3 (Grad Level):
Draw conclusions from this study – about the model type and 
suitability/ deficiencies. Describe what worked and why/ why not. (Min. 6-7 sentences)

Comparing the three models used (Linear Regression, SVM with a linear kernel, and k-NN), the Linear Regression and SVM models performed best, with potential for further improvement through additional outlier detection, removal, and hyperparameter tuning. A regression analysis was more suitable than a classification approach, given the goal of predicting continuous numerical values like Sale Price. The dataset presented significant challenges, including its large size, noise, and dimensionality issues. It was interesting to observe how poorly the Linear Regression model initially performed, which highlighted the importance of preprocessing the data to approach a more normal distribution. The dataset's mix of numerical and categorical factors as well as high variance, highlighted the limitations of simple linear models without sufficient data cleaning. The SVM model demonstrated the potential for higher accuracy but came with a heavy computational cost, which my current hardware could not manage efficiently. From the k-NN model, I learned that high-dimensional datasets reduce its performance significantly. Applying a dimensionality reduction technique, such as PCA, before running the k-NN would have been a good idea to improve its accuracy.


