1.Description of what your team did
  First, we get the data from Kaggle
  Second, we preprocessed the training data
  Third, training data

2.Methods used to accomplish each part
  1)Collect data: Acquire train and test data from Kaggle project(Titanic: Machine Learning from Disaster)
  2)Data preprocessing:
    *Drop name, id, ticket and cabin features
    *Pclass: this feature is already in good condition: only has three values:1, 2 & 3, and no null values
    *Sex: this is a string feature, only two values: female and male, so, we convert it into int:
          female=1, and male=0
    *Age: there are too many null values in age feature, we want to fill it. Instead of using random value
          between mean+-std, inspired by others, we find that people in different age that survived has
          some thing to do with their Pclass and Gender. So instead of guessing age values based on median,
          use random numbers between mean and std, based on sets of Pclass and Gender combinations.
    *Fare: Fare is in a wild range, so we need to band it, and we should decide each band's range, so we
           grouped fare by sorting average survived number
  3)Linear Classify & Cross Validation
    we choose three linear classifiers in this part: logistic regression, perceptron and decision tree,
    and by using 10-fold cross validation, we find that decision tree is the best of the three.
    And due to we only have 891 train point, the classisy result are not very good.

3.Potential difficulties faced
  1)The training data we choose is not that good, we get the data from kaggle, but it has small
    data points, and also too many null values.
  2)Also, this project is the data preprocessing project. Instead of just converting all the feature
    into processible data to make prediction, we need to drop, reshape or create feature for more accurate
    prediction. In order to do this, we have to preview and analyze the data by ploting the train data.
    Moreover, we still need to fill null values in some feature  with reasonable values.

4.Resources used
  1) Data from Kaggle project(Titanic: Machine Learning from Disaster)
  2)sklearn package to finish doing data process, get linear classifer, and sklearn can also used to
    do cross validation.Sklearn is very powerful and sufficient, unless we want to do nerual networks

5.Description of how to run the code in the folder
  Use any python interpreter to “milestones1.py”, change pd.read_csv (About Line 21) path to the path
   of data “train.csv”; Run the code.