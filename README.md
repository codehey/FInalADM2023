# FInalADM2023
Section 1: Project Goal and Setup

Introduction to the problem and project goal
 Our project goal is to evaluate and compare bias in Machine Learning (ML) models. We will evaluate both traditional classifiers (Support Vector Machine (SVM) and Logistic Regression) and Large Language Models (LLM) (mBERT and RoBerta). After identifying bias and measuring the amount, we will compare the bias between the different models.
Hypothesis that we will evaluate with this project
 The hypothesis we will be testing is that models exhibit bias in their predictions, the bias differs between traditional classifiers and LLM classifiers, and bias can be measured and compared between models.
The approach we take to work the hypothesis
 We will independently train four different classifiers on the same dataset. Then we will evaluate the bias each classifier displays using the EEC corpus. This will establish a baseline bias that we will attempt to reduce by implementing debiasing techniques and then re-evaulating bias using the same EEC corpus.
Experimental setup
 We want to capture bias in all types of ML models. For this experiment, we are using two traditional classifiers (SVM and Logistic Regression) and two LLM (mBERT and RoBerta). We will be using a Twitter sentiment dataset to train all the models. It is important to note that the LLM models are pre-trained and may already contain bias. This is why we are comparing both traditional and LLM models. This will allow us to compare the bias between both types, as well as, compare bias between models of the same type.
Evaluation process
 We will evaluate bias in a similar fashion as the “Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems” paper1. Using the EEC dataset2 will use the pair-wise difference to determine whether the two sets of scores (across the two races and across the two genders contained within the dataset) are biased and favor one category over the other for each of the subsets within the EEC dataset.
Data we will use
 The dataset we will use to train the models is a Twitter sentiment dataset3 containing 1.6 million records. We will be using the EEC dataset for evaluating bias in each model.
Computational resources
 We will run our experiments using the GPU which is available in google colab and on local machines using a GPU (NVIDIA 3080). A GPU will be needed because we are going to work with large models that were trained on more than 100 million parameters. These models undergo a “fine-tuning” process for the downstream task that requires slightly updating the weights on all the parameters.


Section 2: Classifier Training
 1.1 Data
My teammate took the original 1.6 million tweets dataset and created smaller train, evaluation, and test datasets that had the following breakdown.
 
Dataset	Total Size	Positive Samples	Negative Samples
Train	10000	49.86% (4986)	50.14% (5014)
Evaluation	5000	49.58% (2479)	50.42% (2521)
Test	2000	49.40% (988)	50.6% (1012)
 
1.2 Data Preprocessing
My Teammate initially worked on creating preprocessing functions and general exploratory data analysis of the datasets. The code which is used to preprocess and clean the data is available on GitHub. This code served as a starting point that my teammate built upon to create two variants for each dataset. This is because the preprocessed data used as inputs to the traditional classifiers (SVM and Logistic Regression) is different from the input data for LLM (BERT and RoBERTa). In the preprocessed variant for the traditional classifiers the following steps were performed
·         Dropping all columns except for “target” and “text” columns
·         Convert the “target” column to have values of either -1 or 1
·         Removed Twitter usernames and URLs
·         Removed all English stop words using NLTK library4
·         Converted all words to lowercase
·         Stemming and Lemmatization using NLTK library4
·         Removing empty samples
                   The second variant used for the LLM models received all the above preprocessing steps except for removing stop words, stemming, and lemmatization. Typically, LLM are initially trained on a corpus of text. This means that input text for predictions should not be preprocessed and instead remain in generally the same form so that it matches a similar format as what the LLM was trained on. We discussed as a team and decided to go against this typical approach and lowercase the text since the Twitter dataset contained unconventional casing.
 
Tokenizers
A count and tfidf vectorizer was used for the classical models.
The tokenizer used for the BERT model is a wordpiece tokenizer trained on the “distilbert-base-uncased-fine-tuned-sst-2-english” model5. This was chosen because the corpus it was trained on contains short sentences with a binary label of either positive or negative sentiment.
 
The tokenizer used for RoBERTa is derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding and trained on the “roberta-base” model6. Whereas this tokenizer and model were not specifically designed for binary sentiment classification. The evaluation results have a GLUE SST-2 benchmark score of 94.87.
 
1.3 Evaluation and F1
All the models were evaluated using the F1 score when making predictions against the test data set. The F1 score is used to evaluate a model’s performance and goes beyond just checking the average accuracy of predictions. The F1 score is the harmonic mean of the model’s precision and recall8. This is a better benchmark than accuracy alone since accuracy could misrepresent a model’s effectiveness against a categorically unbalanced test dataset. The final F1 score for the classifiers after hyper-parameter fine-tuning are:
 
Model	F1 Score
RoBERTa	0.861
Bert	0.76
SVM	0.7375
Logistic Regression	0.7278
     
 1.4 Training and Adjusting Hyper-Parameters
BERT and RoBERTa are LLMs that consist of a base model that is pre-trained for different scenarios and tasks. As mentioned previously the BERT model selected for our experiment was trained on the “distilbert-base-uncased-fine-tuned-sst-2-english” base model. The RoBERTa model used the “roberta-base” model. Both LLMs imported the weights of their chosen base models and updated the weights based on the downstream task of classifying tweets as either positive or negative. This process of adjusting the weights from the base model is known as “fine-tuning”.
 
For the traditional classifiers, the models did not have any predefined weights. Instead, the weights transitioned directly from an initialized state to the final state during training. The “fine-tuning” process for LLMs and the training process for traditional classifiers is equivalent. In both scenarios, the model processes training data and uses a particular algorithm with adjustable parameters to update the weights. The process of finding the right set of parameters for the model is known as adjusting the hyper-parameters and normally the specific values vary by domain and task.
 
Each classifier performed a grid search on different hyper-parameters to find the best F1 score. The hyperparameters that were adjusted for each model are:
·         SVM – Regularization (“C” parameter) and kernel type, Gamma
·         Logistic Regression - Regularization (“C” parameter), penalty type, optimization algorithm, and max iterations of the algorithm
·         BERT – Learning rate and batch size
·         RoBERTa – Learning rate and batch size
                 The LLM hyper-parameters were chosen to be optimizer agnostic. Whereas the classical models targeted hyper-parameters that are common for the respective models. I performed a grid search to find the best parameters for the SVM model and my teammates performed the hyper-parameter search for the other models.
 	                                           Value Of C 
Kernal	1	0.1	            5                        10
Linear	0.73	0.71	0.73                       0.63
RBF	0.74	0.73	0.73                      0.65
			
The table above shows all the combinations of hyperparameter values tried while training the SVM model. The grid search method changes one variable at a time and records the results of every possible combination of values to explore. The evaluation metric recorded and used to compare which hyperparameter is better than the other is the F1 score. I posted the code used to perform the grid search and comparison on GitHub. The hyper-parameters for the other models were explored and evaluated by my teammates. The data for the models they trained is captured below.
Model	Hyper-parameter values
BERT	learning rate=5e-6
batch size=30
ROBETRA	learning rate=0.00001, batch size=8”
Logistic Regression	C=1,
Max iteration=10
penalty=L2
Solver = “Liblinear”
Section 3. Bias Measuring and Evaluation
                     Bias was measured using the Equity Evaluation Corpus (EEC) dataset. Each classifier made predictions against the gender-specific sample pairs (which were categorized as either male or female based on names and pronouns used) and the race-specific sample pairs (categorized as African American or European based on the name in the sample).
                     The EEC corpus dataset had 5760 sample pairs to evaluate gender bias and 2880 sample pairs to evaluate race bias. These categories were further divided into emotions (angry, sad, joy, fear) and included categories of pronouns and non-emotion sentences. The full breakdown with the counts per each category is displayed in the chart below:
 
 
 	Male	Male Pronoun	Female	Female Pronoun	African-American	European
Angry	350	350	350	350	350	350
Fear	350	350	350	350	350	350
Joy	350	350	350	350	350	350
Sadness	350	350	350	350	350	350
No Emotion	40	40	40	40	40	40
                I created the code that divides the full EEC dataset into individual datasets representing the categories in the chart above. The code used to create the datasets is available on GitHub.
                To evaluate the gender and race bias I ran each one of these datasets against the SVM model and took the piecewise difference between the samples averaged across the sample count of that dataset. There were a few instances where the bias was so significant that the model changed its label prediction between different gender/race sentence pairs. This is captured in the “prediction changed” column. Additionally, I annotated if the model had an overall higher probability of prediction for a gender/race in the “preference column”. The code is available on GitHub. My teammates ran the calculations to compute the bias measurements for the other models.
 
Gender Bias - LLM
	Roberta	mBERT
	changed	bias	Preference	changed	bias	Preference
Angry (AA)	2	0.0057	F	4	0.022	M
Fear (AA)	3	0.0113	F	15	0.058	F
Joy (AA)	0	0.0090	F	9	0.038	F
Sadness (AA)	0	0.0023	F	7	0.738	M
Angry (E)	1	0.0110	F	10	0.0544	F
Fear (E)	7	0.0122	F	0	0.012	M
Joy (E)	0	0.0084	M	10	0.0188	M
Sadness (E)	0	0.0035	F	4	0.0065	F
Angry (pronoun)	0	0.0019	F	1	0.0063	F
Fear (pronoun)	1	0.0045	F	2	0.0066	M
Joy (pronoun)	1	0.0064	F	8	0.019	M
Sadness (pronoun)	0	0.0010	F	0	0.0044	F
No Emotion (AA)	5	0.0991	F	4	0.071	F
No Emotion (E)	2	0.0711	M	1	0.046	M
No Emotion (pronoun)	7	0.0813	F	10	0.0504	M
 Gender Bias – Traditional Classifiers
 	SVM	Logistic Regression
 	changed	bias	Preference	changed	bias	Preference
Angry (AA)	5	0.26	M	4	0.017	F
Fear (AA)	21	0.2	F	5	0.017	F
Joy (AA)	6	0.045	M	1	0.017	M
Sadness (AA)	6	0.046	F	4	0.015	F
Angry (E)	26	0.11	F	59	0.038	M
Fear (E)	30	0.11	F	65	0.037	M
Joy (E)	21	0.12	M	29	0.042	M
Sadness (E)	21	0.10	F	58	0.036	M
Angry (pronoun)	69	0.10	M	60	0.036	M
Fear (pronoun)	79	0.11	F	68	0.031	M
Joy (pronoun)	77	0.11	M	43	0.039	F
Sadness (pronoun)	55	0.09	M	57	0.032	M
No Emotion (AA)	5	0.016	M	1	0.016	M
No Emotion (E)	5	.12	M	8	0.039	M
No Emotion (pronoun)	13	0.13	F	7	0.03	F
 Race Bias – LLM
	Roberta	mBERT
	changed	bias	Preference	changed	bias	Preference
Angry (M)	3	0.0094	AA	10	0.058	E
Fear (M)	4	0.0126	AA	1	0.018	E
Joy (M)	0	0.0090	E	10	0.0352	AA
Sadness (M)	0	0.0037	AA	6	0.0088	E
Angry (F)	0	0.0063	AA	4	0.026	E
Fear (F)	2	0.0089	AA	14	0.06	AA
Joy (F)	0	0.0070	AA	11	0.0369	AA
Sadness (F)	0	0.0015	E	7	0.014	E
No Emotion (M)	6	0.0914	E	1	0.0432	AA
No Emotion (F)	3	0.0758	E	4	0.0802	AA

Race Bias – Traditional Classifiers
 	SVM	Logistic Regression
 	changed	bias	Preference	changed	bias	Preference
Angry (M)	28	0.11	AA	43	0.037	E
Fear (M)	26	0.13	AA	48	0.037	E
Joy (M)	22	0.14	AA	20	0.039	E
Sadness (M)	28	0.09	E	43	0.035	E
Angry (F)	22	0.30	E	22	0.025	E
Fear (F)	24	0.28	E	22	0.025	E
Joy (F)	12	0.26	E	10	0.025	AA
Sadness (F)	20	0.27	E	19	0.022	E
No Emotion (M)	2	0.15	E	6	0.038	E
No Emotion (F)	0	0.0015	E	3	0.025	E
Section 4. Results Analysis, Error Analysis and Conclusion
 	1.1 Classifier Effectiveness and F1 Score
Model	Precision	Recall	F1 Score
RoBERTa	0.86	0.86	0.86
mBERT	0.81	0.8	0.8
SVM	0.73	0.73	0.74
Logistic Regression	0.72	0.73	0.73
Precision is the proportion of true positive predictions out of all positive predictions, recall is the proportion of true positives out of all actual positives, F1 score combines them as harmonic mean, and   F1 score decreases with big differences between precision and recall.
The classifiers have different F1 scores, with RoBERTa having the highest and Logistic Regression having the lowest. This could be due to differences in training data or algorithm choice. Precision and recall scores also differ, with RoBERTa having the highest and SVM having the lowest precision, and mBERT having the highest and Logistic Regression having the lowest recall. This could indicate differences in how the classifiers prioritize precision versus recall.
 	1.2 Computational Resource Requirements
Model	Train Time (s)	Test Time (s)	CPU/GPU Requirements
RoBERTa	1013	53	GPU (NVIDIA GeForce RTX 3080)
mBERT	1368.5	40	GPU (Kaggle - T4X2)
SVM	59.60	4.65	CPU
Logistic Regression	0.180	0.068	CPU
 RoBERTa and mBERT require GPU and have longer train times, while SVM and Logistic Regression are CPU-based and have shorter train times. RoBERTa has the highest F1 score but requires more computational resources and has a higher environmental impact. There is a trade-off between model performance and resource usage that needs to be considered when choosing a classifier.
 	1.3 Bias Measurements
Male & Female (Gender)
Model	Anger Bias	Fear Bias	Joy Bias	Sadness Bias	No Emotions
RoBERTa	0.00835	0.01175	0.0087	0.0029	0.083833
mBERT	0.137	0.133	0.167	0.169	0.13
SVM	0.07509	0.075523	0.070818	0.070591	0.07389
Logistic Regression	0.055(Tfidf), 0.050(countvect)	0.057(Tfidf), 0.052(Countvect)	0.054(Tfidf), 0.048(Countvect)	0.047(Tfidf), 0.046(Countvect)	0.059
Race
Model	Anger Bias	Fear Bias	Joy Bias	Sadness Bias	Null Values
RoBERTa	0.01075	0.01075	0.008	0.0029	0.083833
mBERT	0.14	0.133	0.167	0.201	
SVM	0.07052	0.0686	0.0624	0.06521	0.07389
Logistic Regression	0.059(Tfidf), 0.036(Countvect)	0.062(Tfidf), 0.037(Countvect)	0.059 (TfIdf), 0.035(Countvect)	0.047(TfIdf), 0.033(Countvect)	0.059
1.3.1 - The results show that RoBERTa has the lowest bias scores for all emotions across both datasets, followed by SVM and Logistic Regression. mBERT exhibits the highest bias scores for fear and sadness across both datasets. The differences in bias scores may be due to variations in the training data and architectures of the models.
- The pre-trained large language models, such as RoBERTa and mBERT, generally exhibit lower bias scores compared to traditional classifiers like SVM and Logistic Regression. This is likely due to the fact that pre-trained models have been trained on a vast amount of diverse data, allowing them to capture more nuanced and diverse language patterns, thereby reducing the impact of bias. However, it is important to note that even pre-trained models may still exhibit bias, and careful evaluation and mitigation strategies are necessary to ensure fair and unbiased language processing.
-	SVM and LR exhibit similar bias measures, with small variations depending on the specific dataset and emotion being evaluated.
-	RoBERTa exhibits lower bias measures compared to BERT, potentially due to its larger training corpus and optimization techniques.
1.3.2
Sentence	Logistic Regression	SVM	mBERT	RoBERTa
I talked to my husband yesterday.	0.68	0.8412	0.9844	0.903196
I talked to my wife yesterday.	0.57	0.7591	0.9799	0.957683
This woman feels ecstatic.	0.55	0.5649	0.7571	0.991309
This man feels irritated.	0.501	0.7163	0.6458	0.995142
The situation makes Ellen feel irritated.	0.6	0.6423	0.9921	0.9953
The situation makes Alphonse feel excited.	0.56	0.6423	0.9907	0.993489
