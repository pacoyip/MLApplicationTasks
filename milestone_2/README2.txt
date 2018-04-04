1.Description of what your team did
  Add four Gaussain Process Classifier with 4 different kernel:RBF, DotProduct, DotProduct**2, Constant

2.Methods used to accomplish each part
    1)Gaussain Processor Classifier & Cross Validation
      we choose Gaussain Process Classifier with 4 different kernel in this part: RBF, DotProduct,
      Constant and by using 10-fold cross validation, we find that Gaussain Process Classifier with
      DotProduct^2 kernel is the best with negative-log-loss=6.63+-1.86
    2)In the error type, we use make_score(log_loss) to compute the negative-log-loss

3.Potential difficulties faced
  we want a kernel which perform not that good to show that other kernels are powerful kernels, so, we
  choose constant kernel as the control kernel, RBF and DotProduct as experimental kernel

4.Resources used
  1) Data from Kaggle project(Titanic: Machine Learning from Disaster)
  2) Get Gaussain Process Classifier, kernel and cross validation from sk-learn package

5.Description of how to run the code in the folder
  Use any python interpreter to “milestones2.py”, change pd.read_csv (About Line 28) path
  to the path of data “train.csv”; Run the code.

6.Analtsis of each kernel:
  1)In milestones2, we use four kinds of kernels in GP: RBF, square of DotProduct, and constant kernel
    RBF: RBF kernel is a similarity-based method, which function is :
       k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)
       it is infinitely differentiable, which implies that GPs with this kernel as covariance function
       have mean square derivatives of all orders, and are thus very smooth.
       And due to RBF kernel is a very powerful kernel, we use it in our project
    DotProduct^2 : this kernel is k(x_i, x_j) = sigma_0 ^ 2 + x_i cdot x_j, and is non-stationary and
       can be obtained from linear regression by putting N(0, 1) priors on the coefficients of
       x_d (d = 1, . . . , D) and a prior of N(0, sigma_0^2) on the bias.
  2)Compare these two, the The latter works better than the former， which is a little bit like the
    example in the sk-learn tutorial:
    http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_xor.html#...
         ...sphx-glr-auto-examples-gaussian-process-plot-gpc-xor-py
    and the reason we think is:
    Compared are a stationary, isotropic kernel (RBF) and a non-stationary kernel (DotProduct).
    On our dataset, the DotProduct kernel obtains considerably better results because the class-boundaries are
    linear and coincide with the coordinate axes. In general, stationary kernels often obtain better results.
  3) And of course constant kernel give the worst result due to is give constant value in the kernel matrix.


7.Result of each method:
  Negative-log-loss of logreg after 10fold-val: 7.29 (+/- 2.01)
  Negative-log-loss of percep after 10fold-val: 8.96 (+/- 3.48)
  Negative-log-loss of DCtree after 10fold-val: 6.70 (+/- 3.26)
  Negative-log-loss of GP+RBF after 10fold-val: 6.82 (+/- 1.52)
  Negative-log-loss of GP+D^2 after 10fold-val: 6.63 (+/- 1.86)
  Negative-log-loss of GP+Cst after 10fold-val: 13.26 (+/- 0.20)