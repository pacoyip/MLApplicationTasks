1.Description of what your team did
  Add four Gaussain Process Classifier with 4 different kernel:RBF, DotProduct, DotProduct**2, Constant

2.Methods used to accomplish each part
    1)Gaussain Processor Classifier & Cross Validation
      we choose Gaussain Process Classifier with 4 different kernel in this part: RBF, DotProduct, DotProduct**2, Constant and by using 10-fold cross validation, we find that Gaussain Process Classifier with DotProduct^2 kernel is the best with negative-log-loss=6.63.
    2)In the error type, we use make_score(log_loss) to compute the negative-log-loss

3.Potential difficulties faced
  Cannot figure out why Gaussain Process Classifier with kernel DotProduct^2 performed a little bit better than with kernel DotProduct

4.Resources used
  1) Data from Kaggle project(Titanic: Machine Learning from Disaster)
  2) Get Gaussain Process Classifier, kernel and cross validation from sk-learn package

5.Description of how to run the code in the folder
Use any python interpreter to “milestones2.py”, change pd.read_csv (About Line 28) path to the path of data “train.csv”; Run the code.

6.Result of each method:
  Negative-log-loss of logreg after 10fold-val: 7.29 (+/- 2.01)
  Negative-log-loss of percep after 10fold-val: 8.96 (+/- 3.48)
  Negative-log-loss of DCtree after 10fold-val: 6.70 (+/- 3.26)
  Negative-log-loss of GP+RBF after 10fold-val: 6.82 (+/- 1.52)
  Negative-log-loss of GP+Dot after 10fold-val: 7.21 (+/- 1.99)
  Negative-log-loss of GP+D^2 after 10fold-val: 6.63 (+/- 1.86)
  Negative-log-loss of GP+Cst after 10fold-val: 13.26 (+/- 0.20)