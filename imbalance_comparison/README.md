# Data Classification in Medical/Healthcare: Classification Methods Comparison under Class Imbalance #
<br>
This paper mainly focuses on the performance comparison of several imbalance classification algorithms. This repository includes the code of some of these algorithms, namely, DDAE, Iterative Metric Learning (IML), Self-paced Ensemble Classifier (SPE) and CAdaMEC. Other algorithms used for the comparison can be found on [imblearn](https://pypi.org/project/imblearn/) and [costcla.models](https://pypi.org/project/costcla/).

**DDAE** is implemented based on [1]

**Iterative Metric Learning (IML)** is implemented based on [2]

The work of **Self-paced Ensemble Classifier (SPE)** [3] is from [SPE](https://github.com/ZhiningLiu1998/self-paced-ensemble?utm_source=catalyzex.com#miscellaneous)
<br>
<pre>
    @inproceedings{
	liu2020self-paced-ensemble,
    title={Self-paced Ensemble for Highly Imbalanced Massive Data Classification},
    author={Liu, Zhining and Cao, Wei and Gao, Zhifeng and Bian, Jiang and Chen, Hechang and Chang, Yi and Liu, Tie-Yan},
    booktitle={2020 IEEE 36th International Conference on Data Engineering (ICDE)},
    pages={841--852},
    year={2020},
    organization={IEEE}
</pre>

The work of **CAdaMEC** [4] is from [Calibrated AdaMEC](https://mloss.org/software/view/671/)
<pre>
	@Article{Nikolaou2016,
	author="Nikolaou, Nikolaos
	and Edakunni, Narayanan
	and Kull, Meelis
	and Flach, Peter
	and Brown, Gavin",
	title="Cost-sensitive boosting algorithms: Do we really need them?",
	journal="Machine Learning",
	year="2016",
	volume="104",
	number="2",
	pages="359--384",
	abstract="We provide a unifying perspective for two decades of work on cost-sensitive Boosting algorithms. When analyzing the literature 1997--2016, we find 15 distinct cost-sensitive variants of the original algorithm; each of these has its own motivation and claims to superiority---so who should we believe? In this work we critique the Boosting literature using four theoretical frameworks: Bayesian decision theory, the functional gradient descent view, margin theory, and probabilistic modelling. Our finding is that only three algorithms are fully supported---and the probabilistic model view suggests that all require their outputs to be calibrated for best performance. Experiments on 18 datasets across 21 degrees of imbalance support the hypothesis---showing that once calibrated, they perform equivalently, and outperform all others. Our final recommendation---based on simplicity, flexibility and performance---is to use the original Adaboost algorithm with a shifted decision threshold and calibrated probability estimates.",
	issn="1573-0565",
	doi="10.1007/s10994-016-5572-x",
	url="http://dx.doi.org/10.1007/s10994-016-5572-x"
	}
</pre>

The work of MetaCost is from [MetaCost](https://github.com/Treers/MetaCost) which based on [5]
# Install #
- - -
<br>
Our implementation requires following dependencies:
	

- [python](https://www.python.org/)(>=3.6)
- [numpy](https://numpy.org/)(>=1.19.5)
- [imblearn](https://pypi.org/project/imblearn/)(>=0.7.0)
- [scikit-learn](https://scikit-learn.org/stable/)(>=0.23.1)
- [joblib](https://pypi.org/project/joblib/)(>=0.16.0)
- [costcla.models](https://pypi.org/project/costcla/)(>=0.6)
<br>

# Miscellaneous #
- - -
This repository contains:
<br>

- Implementation of DDAE and IML
- Implementation of SPE and CAdaMEC
- Example Datasets used for experiments

# Usage #
- - -
## DDAE ##
### Documentation ###


|    Parameters    | Description |
| ---------- | --- |
| lb_column |  target name |
| maj_label |  the label of majority |
|min_label	|the label of minority |
|n_neighbors|number of the neighbors |
|max_iter |maximal iteration|
|cost_ratio|ratio between the cost of false negative and that of the false positive. **Default value = 2**|
|weight_loss_pull|relative weight between pull and push terms in large margin nearest neighbor algorithm. **Default value = 0.2**|
|unstable_ratio|Used to determine wheter the awa component should be adapted or not. **Default value = 0.2**|

|    Methods    | Description |
| ---------- | --- |
| `fit(self, X, y)` |  Build a DDAE from the training set (X, y). |
| `predict(self, X)` |  Predict class for X. |
|`predict_proba(self, X)`	|Predict class probabilities for X. |



## Iterative Metric Learning (IML) ##
### Documentation ###
|    Parameters    | Description |
| ---------- | --- |
| target_name |  target name |
| n_neighbors |  number of the neighbors |
|regularization	|elative weight between pull and push terms in large margin nearest neighbor algorithm|
|base_classifier|Classifier used for final classification |
|max_iter |maximal iteration for LMNN|
|label_tupel|label of negative and positive in tuple. <br> For example (0, 1)|
|top_positive_number|number of selected positive nearest neighbors|
|top_negative_number|number of selected negative nearest neighbors|
|matching_ratio|Used to determine wheter the selected sub training set is stable or not|

|    Methods    | Description |
| ---------- | --- |
| `fit(self, X, y)` |  Build a DDAE from the training set (X, y). |
| `predict(self, X)` |  Predict class for X. |
|`predict_proba(self, X)`	|Predict class probabilities for X. |

## Example ##
**The label of samples processed by DDAE and IML should be ('0'. '1'). But for SPE and CAdaMEC, it should be (0, 1).**

<pre class="prettyprint lang-javascript">
	
	tup = (negative, positive)
	y_train = y_train.map({tup[0]: '0', tup[1]: '1'})
	y_test = y_test.map({tup[0]: '0', tup[1]: '1'})
	cost_fn = 42 	# cost for false negative, default value is the imbalance ratio of the current dataset

	# DDAE
	model = MainDDAE(lb_column=label, maj_label='0', min_label='1', n_neighbors=5, max_iter=50)
	
	model.fit(X_train, y_train)
	predict_ddae = model.predict(X_test)
	predict_proba = model.predict_proba(X_test)
	evaluate_metrics.ev_me(y_test, predict_ddae, ['0', '1'])
	evaluate_metrics.draw_apr_c(y_test, predict_proba, '1', 'DDAE')
	evaluate_metrics.draw_roc_c(y_test, predict_proba, '1', 'DDAE')

	# IML
	model = IterativeMetricLearning(target_name=lb,
	                                n_neighbors=3,
	                                regularization=0.2,
	                                base_classifier=KNeighborsClassifier(n_neighbors=3),
	                                top_negative_number=5,
	                                top_positive_number=6,
	                                label_tupel=('0', '1'))
	model.fit(X_train, y_train)
	predict_iml = model.predict(X_test)
	
	predict_proba = model.predict_proba()
	
	evaluate_metrics.ev_me(y_test, predict_iml, ['0', '1'])
	evaluate_metrics.draw_apr_c(y_test, predict_proba, '1', 'IML')
	evaluate_metrics.draw_roc_c(y_test, predict_proba, '1', 'IML')
	
	# SPE
	def absolute_error(y_true, y_pred):
		"""Self-defined classification hardness function"""
		return np.absolute(y_true - y_pred)
	
	base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=9,
	                                        min_samples_split=2,
	                                        splitter='random')
	
	model = SelfPacedEnsemble(base_estimator=base_estimator, hardness_func=absolute_error,
	                          n_estimators=10).fit(np.array(X_train), np.array(y_train))
	predict_spe = model.predict(np.array(X_test))
	y_score = model.predict_proba(np.array(X_test))
	
	evaluate_metrics.ev_me(y_test, predict_spe, [0, 1])
	evaluate_metrics.draw_apr_c(y_test, y_score, 1, 'self-paced ensmeble')
	evaluate_metrics.draw_roc_c(y_test, y_score, 1, 'self-paced ensmeble')

	# CAdaMEC
	base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=15, max_leaf_nodes=9,
	                                        min_samples_split=2,
	                                        splitter='random')
	
	cost_fp = 1
	
	CaAdaMEC = CalibratedAdaMECClassifier(base_estimator, n_estimators, cost_fp, cost_fn)
	CaAdaMEC.fit(np.array(X_train), np.array(y_train))
	predict_ca = CaAdaMEC.predict(np.array(X_test))
	y_score_ca = CaAdaMEC.predict_proba((np.array(X_test)))
	
	evaluate_metrics.ev_me(y_test, predict_ca, [0, 1])
	evaluate_metrics.draw_apr_c(y_test, y_score_ca, 1, 'CAdaMEC')
	evaluate_metrics.draw_roc_c(y_test, y_score_ca, 1, 'CAdaMEC')
	
	# MetaCost
	cost_fp = 1
    cost_matrix = np.array([[0, cost_fn], [cost_fp, 0]])

    model = MetaCost(pd.concat([X_train, y_train], axis=1), DecisionTreeClassifier(), cost_matrix).fit(label, 2)

    predict_meta = model.predict(X_test)
    y_score_meta = model.predict_proba(X_test)

    evaluate_metrics.ev_me(y_test, predict_meta, [0, 1])
    evaluate_metrics.draw_apr_c(y_test, y_score_meta, 1, 'MetaCost')
    evaluate_metrics.draw_roc_c(y_test, y_score_meta, 1, 'MetaCost')

</pre>
The output for a single algorithm can be generated by `evaluate_metrics.py`, which looks like
<pre>
# Confusion Matrix
[[3249   23]
 [  46   37]]
# Evaluation Metrics
recall 0.4457831325301205
precision 0.6166666666666667
g_mean 0.6653191500255327
f1 0.5174825174825175
AUCPRC 0.4882785333543856
AUC 0.8589068989896015
</pre>

# References #
- - -
|    #    | Reference |
| ---------- | --- |
| [1] |  J. Yin, C. Gan, K. Zhao, X. Lin, Z. Quan, and Z.-J. Wang, “A novel model for imbalanced data classification,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04,2020, pp. 6680–6687. |
| [2] |  N. Wang, X. Zhao, Y. Jiang, Y. Gao, and K. BNRist, “Iterative metric learning for imbalance data classification.” in IJCAI, 2018, pp. 2805–2811. |
|[3]	|Z. Liu, W. Cao, Z. Gao, J. Bian, H. Chen, Y. Chang, and T.-Y. Liu, “Self-paced ensemble for highly imbalanced massive data classification,” in 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020, pp. 841–852.|
|[4]	|N. Nikolaou, N. Edakunni, M. Kull, P. Flach, and G. Brown, “Cost-sensitive boosting algorithms: Do we really need them?” Machine Learning, vol. 104, no. 2, pp. 359–384, 2016. |
|[5] | P. Domingos, “Metacost: A general method for making classifiers cost-sensitive,” in Proceedings of the fifth ACM SIGKDD international conference on Knowledge discovery and data mining, 1999, pp. 155–164.|