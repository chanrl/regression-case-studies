# regression-case-studies

## Objectives
- Explore data pulled from various Kaggle competitions
- Clean and standardize data for future processing
- Apply various regression techniques
- Create and optimize a prediction model

## Background
Using data from Kaggle's [NYC rental listing inquiries competition](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries) and the [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data) auction price prediction competition, follow the cleaning and modeling guideline in the [Mechanics of Machine Learning](https://mlbook.explained.ai/) book to create a prediction model for the two datasets.

The first model should be able to predict the rent on a new rental listing in NYC, and the second model would predict the auction price for a bulldozer given certain features and details.

## Conclusion
For the NYC Rental listing, the model was able to achieve a validation R-squared score of .87 with Random Forest Regressor. I was able to practice more exploration, denoising and cleaning techniques. I also learned more about feature engineering, getting a baseline model running, encoding categorical variables, extracting features from strings, and synthesizing numerical features.

For the Blue Book for Bulldozers, the model was able to achieve a .42 Root Mean Squared Log Error using Ridge Regression. The Best RMSLE in the competition was .23, and simply guessing the median auction price for all entries in the test set would get you a RMSLE of .7. The most important techniques I learned in this one has to do with cleaning large databases. Previous functions I wrote would take a couple seconds per 100 entries, and after revising how to write functions to clean and categorize specified dataframe columns, it would only take a couple seconds for the entire database (over 100,000 row entries.)

Additionally, I was able to practice how to select certain features for the final model by importance. Even though the validation score may not increase, the new categories that you clean and create may still be useful later as you select for the best features to keep.