# ML-Python Practice 1: Real Estate Prediction (Linear Regression)

This project is referred from below link for study purpose. Credit to the owner:
https://thecleverprogrammer.com/2023/12/11/real-estate-price-prediction-using-python/

From this practice, I learnt;
1-To clean and prepare the data by handling missing values, removing outliers, and converting categorical variables to numerical representations.
2-To explore and visualize the data to gain insights into its distribution, correlations and patterns.
3-To choose appropriate machine learning algorithm or predictive model for the task.
4-To train the selected model on the training data, optimizing its parameters to make accurate predictions.

//Histograms of Real Estate Data.png//
- Provide insights into the distribution of each variable.
Significant analysis:
1-There are more houses located nearby the MRT station
2-Number of convenience stores has an obvious relationship to the concentration of houses in an area
3-The concentration of properties in lower price range is obvious compared to the higher price-range properties

//Scatter Plots with House Price of Unit Area.png//
- To explore the relationship between these variables with the house price.
Observation:
1-House age vs. price: No strong linear relationship, it appears that very new and old properties might have higher prices.
2-Distance to MRT station vs. price: Clear trend showing that as the distance increases, the price decreases
3-Number of convenience stores vs. price: Positive relationship; Houses with more convenience stores in the vicinity tend to have higher prices
4-Latitude vs. price: Not a strong linear relationship, but there seems to be pattern where certain latitude corresponds to higher or lower prices. It could be indicative of specific neighbourhood being more desirable.

//Correlation Matrix.png//
- To quantify the relationships between these variables.
Observation:
- Distance to MRT station and number of covenience stores has a strong/moderate +/-ve correlation with house price

//Linear Regression model to predict the real estate prices - Actual vs. Predicted House Prices.png//
- Diagonal dashed line represents where the actual and predicted values would be equal.
Observation:
1-Many points are close to the diagonal line, suggesting that the model makes reasonably accurate predictions for a significant portion of the test set.
2-Some points are further from the line, indicating areas where the model's predictions deviate more significantly from the actual values
