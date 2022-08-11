# factor-analysis-through-python
Factor analysis is one of the unsupervised machine learning algorithms which is used for dimensionality reduction. This algorithm creates factors from the observed variables to represent the common variance i.e. variance due to correlation among the observed variables.

BARTLETT’S TEST OF SPHERICITY
Bartlett’s test checks whether the correlation is present in the given data. It tests the null hypothesis (H0) that the correlation matrix is an Identical matrix. The identical matrix consists of all the diagonal elements as 1. So, the null hypothesis assumes that no correlation is present among the variables.

We want to reject this null hypothesis because factor analysis aims at explaining the common variance i.e. the variation due to correlation among the variables. If the p test statistic value is less than 0.05, we can decide that the correlation is not an Identical matrix i.e. correlation is present among the variables with a 95% confidence level.


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericitychi2,p = calculate_bartlett_sphericity(dataframe)
print("Chi squared value : ",chi2)
print("p value : ",p)#OUTPUT:Bartlett Sphericity TestChi squared value : 4054.19037041082
p value : 0.0



Read the dataset using pandas and store the dataset in a dataframe. We have stored the dataset in a dataframe named ‘dataset’. Just simply pass the ‘dataset’ through the calculate_bartltett_sphericty function, it will test the null hypothesis and will return the chi-squared value and the p test statistic. Since the p test statistic is less than 0.05, we can conclude that correlation is present among the variables which is a green signal to apply factor analysis.

KMO Test measures the proportion of variance that might be a common variance among the variables. Larger proportions are expected as it represents more correlation is present among the variables thereby giving way for the application of dimensionality reduction techniques such as Factor Analysis. KMO score is always between 0 to 1 and values more than 0.6 are much appreciated. We can also say it as a measure of how suited our data is for factor analysis.


from factor_analyzer.factor_analyzer import calculate_kmokmo_vars,kmo_model = calculate_kmo(dataset)
print(kmo_model)#OUTPUT:
KMO Test Statistic 0.8412492848324344


Just pass the dataframe which contains information about the dataset to the calculate_kmo function. The function will return the proportion of variance for each variable which is stored in the variable ‘kmo_vars’ and the proportion of variance for the whole of our data is stored in ‘kmo_model’. we can see that our data has an overall proportion of variance of 0.84. It shows that our data has more correlation and dimensionality reduction techniques such as the factor analysis can be applied.

DETERMINING THE NUMBER OF FACTORS
The number of factors in our dataset is equal to the number of variables in our dataset. All the factors are not gonna provide a significant amount of useful information about the common variance among the variables. So we have to decide the number of factors. The number of factors can be decided on the basis of the amount of common variance the factors explain. In general, we will plot the factors and their eigenvalues.

Eigenvalues are nothing but the amount of variance the factor explains. We will select the number of factors whose eigenvalues are greater than 1.

But why should we choose the factors whose eigenvalues are greater than 1? The answer is very simple. In a standard normal distribution with mean 0 and Standard deviation 1, the variance will be 1. Since we have
standard scaled the data the variance of a feature is 1. This is the reason for selecting factors whose eigenvalues(variance) are greater than 1 i.e. the factors which explain more variance than a single observed variable.

from factor_analyzer import FactorAnalyzerfa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=dataframe.shape[1])fa.fit(dataframe)
ev,_ = fa.get_eigenvalues()plt.scatter(range(1,dataframe.shape[1]+1),ev)
plt.plot(range(1,dataframe.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()
