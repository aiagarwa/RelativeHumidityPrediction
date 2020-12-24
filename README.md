# RelativeHumidityPrediction  
Group Assignment on prediction of Relative Humidity using data collected from Buoys located around Ireland.   
All the main Code Files used at the end are available in the *main* branch. Also the rest of the individual work is done in the respective *named branches*.  
The main data file is the HumidityDataset.csv  
The Unseen Live Data File from 21st of December is UnseenTest. This Data is Available at  https://www.met.ie/latest-reports/observations .   
The rest of the csv files are imputed data files Done using various Imputation techniques.    
  


## Record of the Work Done By Sherwin Mascarenhas  
## The Main Code Files for the Neural Network is Available on the Main Branch

There are a total of 5 Python Files.      
*FinalImputedDataFeatureSet1*  --> Contains the regression model that was done on Feature Set 1 mentioned in the Report using the MLP model     
*FinalImputedDataFeatureSet2*  --> Contains the regression model that was done on Feature Set 2 mentioned in the Report using the MLP model   
*FinalImputedDataFeatureSet3*  --> Contains the regression model that was done on Feature Set 3 mentioned in the Report using the MLP model   
*FinalRegressionNotImputedFeatureSet3*      --> Non Imputed Data with Feature Set 3 was created to check whether the imputed Data performs well. According to the Cross Validation it does  
*FinalClassificationModel*                  --> This Classification Model was created to try and Improve the Score of the Model without Dew Point. This was Scraped since our model started doing well with the Imputed Data.  
*InitialDatacleaningFrameworkUsedInAllFiles*   ---> The Inital Data Cleaning Base that Was used in most the files.
*RandomForestDatSelection*                    --->  This was a Feature Selection notebook created. 


## Record of the Work Done by Chitransh  

The Feature selection and the Baseline Models are done by Chitransh     
His Code is available in branches that start with his name.  


## Record of the Work Done By Aishwarya  
## The Main Code Files for the Data Imputation and Knn Regression is Available on the Main Branch

There are 7 python Files.

*DataImputation_V5.ipynb* --> Contains the code for Data Imputation for all three techniques (Mean, Most Frequent, Median)

*kNNModel_ImputedData_Iterative.ipynb* --> Contains the code for checking how Data Imputation using Iterative Imputer works on kNN Regressor

*kNNModel_ImputedData_mean.ipynb* --> Contains the code for checking how Data Imputation using Mean Imputer works on kNN Regressor

*kNNModel_ImputedData_most_frequent.ipynb* --> Contains the code for checking how Data Imputation using Iterative Imputer works on kNN Regressor

*kNNModel_RegressionImputedData_FS1.ipynb* --> Contains the code for the regression model using kNN Regressor on Feature Set 1 that was mentioned in the Report.

*kNNModel_RegressionImputedData_FS2.ipynb* --> Contains the code for the regression model using kNN Regressor on Feature Set 2 that was mentioned in the Report.

*kNNModel_RegressionImputedData_FS3.ipynb* --> Contains the code for the regression model using kNN Regressor on Feature Set 3 that was mentioned in the Report.