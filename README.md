# Credit Card Fraud Detection

## About Credit Fraud Dataset :
- From [Kaggle (Credit Card Fraud Detection)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Highly unbalanced data (492 frauds out of 284,807 transactions)


### Download Dataset
```
$ cd ..  
$ mkdir data  
$ cd data  
$ kaggle datasets download -d mlg-ulb/creditcardfraud 
$ unzip creditcardfraud.zip
$ rm creditcardfraud.zip
```

### Columns
- **Input variables**
    - 28 numerical input variables V which are the result of a PCA transformation
    - 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. (172792.0 seconds = about 48 hours = about 2 days)
    - 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning
- **Target Variable**
    - 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
