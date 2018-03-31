1.Description of what your team did
  Add four Gaussain Process Classifier with 4 different kernel:RBF, DotProduct, DotProduct**2, Constant

2.Methods used to accomplish each part
    Gaussain Processor Classifier & Cross Validation
    we choose Gaussain Process Classifier with 4 different kernel in this part: RBF, DotProduct, DotProduct**2, Constant and by using 10-fold cross validation, we find that Gaussain Process Classifier with DotProduct^2 kernel is the best with 0.81 score.

3.Potential difficulties faced
  Cannot figure out why Gaussain Process Classifier with kernel DotProduct^2 performed a little bit better than with kernel DotProduct

4.Resources used
  1) Data from Kaggle project(Titanic: Machine Learning from Disaster)
  2) Get Gaussain Process Classifier, kernel and cross validation from sk-learn package

5.Description of how to run the code in the folder
Use any python interpreter to “milestones2.py”, change pd.read_csv (About Line 28) path to the path of data “train.csv”; Run the code.