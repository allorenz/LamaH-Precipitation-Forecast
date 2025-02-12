## Motivation

In the era of AI and ML, particularly in the context of climate change, applications range from predicting power demand in specific regions, forecasting future carbon emissions, to weather prediction. This project focuses on predicting precipitation for the next day. To achieve this, various machine learning algorithms (Linear Regression, Random Forest, Gradient Boosting, Neural Network) are trained on comprehensive time-series weather data from the LamaH dataset. LamaH (**La**rge-Sa**m**ple D**a**ta for **H**ydrology and Environmental Sciences) contains a combination of meteorological time series and hydrologically relevant attributes from over 859 catchments in Central Europe, covering Austria, Germany, the Czech Republic, Switzerland, Slovakia, Italy, Liechtenstein, Slovenia, and Hungary.

## Analysis

![average precipitation](output/avg_prec_by_month.png)

![average precipitation per month](output/avg_prec_by_month.png)

The precipitation trend remains relatively stable between 1981 and 2019, although some outliers are present. Additionally, it is evident that the central region of Europe exhibits a clear seasonal pattern, with significantly higher precipitation in the summer months compared to winter.

![cyclic encoding](output/avg_prec_by_month.png)

To enhance the dataset, the seasonality has been cyclically encoded, ensuring that data for the winter months (January, February, and March) is treated differently from the summer months.

## Results

![bar chart](output/rmse.png)