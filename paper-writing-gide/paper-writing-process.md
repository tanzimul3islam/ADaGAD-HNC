A Structured Template & Writing Guide for a
Conference Paper on a Classification Problem
Machine Learning / Deep Learning Track — Section-by-section blueprint, guiding questions,
sentence patterns, and ready-to-fill tables.
This document is a writing scaffold, not a finished paper. Each section explains what to write, why reviewers
expect it, and gives sentence patterns and tables you can adapt to your own dataset and results. Replace
every bracketed placeholder [like this] with your actual content.

1. Title
The title should be specific enough that a reader immediately understands the task, the method family, and
(if relevant) the domain/dataset. Aim for 12–18 words.
Formula: [Task/Problem] using [Method/Architecture]: [Domain-specific angle or dataset]
Example: "Skin Disease Classification Using a Hybrid CNN–Random Forest Framework: A
Feature-Importance-Based Approach"
(cid:127) Avoid vague titles such as "A Study on Classification Using Machine Learning."
(cid:127) Mention the specific algorithm(s) only if they are central to the contribution.
(cid:127) Keep acronyms to a minimum; spell out the first use.
2. Abstract (150–250 words)
The abstract is a miniature version of the entire paper. Reviewers often decide whether to read further based
on this paragraph alone. Write it last, after the rest of the paper is finished, and keep it to a single paragraph
with no citations.
Component Purpose Guiding sentence starter
Problem State the classification task and why "[Disease/task] classification remains challenging due
it is difficult. to [X, Y, Z]."
Objective State the study's goal in one "This study aims to develop a [model type] capable of
sentence. [goal]."
Approach Name the ML/DL algorithms and "We propose a framework that combines [technique
pipeline briefly. 1] with [technique 2]."
Dataset Describe the data source, size, and "The model was trained and evaluated on a dataset
classes. of [N] samples across [k] classes."
Results Report the headline metric(s). "The proposed model achieved an accuracy of
[XX]%, outperforming [baseline] by [Y] points."
Contribution State why this matters to the field. "These results demonstrate the potential of
[approach] for [real-world application]."
Tip: write six one-sentence answers to the six rows above, then stitch them into one flowing paragraph — that alone
will usually land close to 150–200 words.
3. Keywords
List 4–6 terms, ordered from most general to most specific, separated by semicolons. Use terms that authors
of related work would actually search for.
Example: Skin disease classification; deep learning; feature selection; convolutional neural network; random forest;
medical image analysis

4. Introduction (1.5–2 pages)
The introduction moves from broad context to your specific contribution, in a funnel shape. Use the six-part
flow below as an outline; each part is typically one short paragraph.
(cid:127) 4.1 Background of the problem — Introduce the domain and why the task matters (e.g., public health
burden, economic impact, safety concern).
(cid:127) 4.2 Contributing factors — Explain what makes the problem hard: class imbalance, visual/data similarity
between classes, noisy labels, limited samples, subjectivity of manual diagnosis, etc.
(cid:127) 4.3 Statistical evidence of the problem — Cite concrete numbers (prevalence, incidence rate,
misdiagnosis rate, economic cost) with a source, to justify urgency.
(cid:127) 4.4 Traditional / existing approaches — Briefly summarize how the problem has historically been
addressed (manual inspection, classical ML, rule-based systems) and their shortcomings.
(cid:127) 4.5 Your approach — Introduce, in general terms, the method(s) you propose and how they address the
shortcomings above.
(cid:127) 4.6 Contributions — Close with a short, numbered list of concrete contributions.
Example sentence pattern: "[Disease] affects approximately [X million] people worldwide each year [ref], making
early and accurate diagnosis critical to reducing [outcome]. However, [factor 1] and [factor 2] make manual diagnosis
time-consuming and prone to inter-observer variability. Traditional approaches based on [method] have achieved
[result], but suffer from [limitation]. In this study, we propose [approach name], which [core idea], to address these
limitations."
Suggested contribution list format
The key contributions of this study are as follows:
1. We propose a [novel/hybrid] framework that [what it does].
2. We evaluate the framework on [dataset], comprising [N] samples across [k] classes.
3. We compare [n] machine/deep learning models and identify [best model] as the top performer, achieving [XX]%
accuracy.
4. We deploy the trained model as a [web app / API] to demonstrate real-world applicability.
Length guide: with 11–12 pt font and standard margins, 1.5–2 pages is roughly 700–1000 words.

5. Literature Review (10–15 papers)
Open with 2–3 sentences framing how the field has evolved (e.g., from classical ML to deep learning, from
single-modality to hybrid pipelines). Then discuss papers in thematic or chronological clusters — never as an
unconnected  list  of  one-line  summaries.  For  each  paper,  report:  dataset  used,  preprocessing/feature
engineering, model(s), and the reported metric.
5.1 Sentence patterns for summarizing a study
[Author(s)] [presented / proposed / developed / constructed] a [machine/deep learning]-based [architecture/framework]
for [task]. The study used a dataset of [N] [subjects/images], consisting of [k] classes. During preprocessing, the
authors applied [technique(s)], and [m] features were selected/ranked using [method] based on their importance. For
classification, a [Model A]/[Model B] pipeline was used, ultimately achieving an accuracy of [XX]% [ref].
Example  sentence  pattern:  "Silva  et  al.  proposed  a  machine  learning-based  framework  for  skin  disease
classification. Their dataset included 6,080 skin-patient records. In the preprocessing stage, they applied [technique],
and 58 features were ranked using BorutaShap according to their importance and relevance. The Random Forest
model, trained on the selected features, achieved an accuracy of 86% [18]."
Vary  your  verbs  across  the  section  (proposed,  developed,  introduced,  investigated,  constructed,  designed,
implemented) so the review does not read like a repeated template.
5.2 Summary Table 1 — Comparative overview of reviewed studies
| Reference | Dataset |     | Method |     | Accuracy |
| --------- | ------- | --- | ------ | --- | -------- |
Silva et al. [18] [Dataset name] BorutaShap, Random Forest 86%
| [Author et al.] [ref] | [Dataset name] |     | [Method] |     | [XX%] |
| --------------------- | -------------- | --- | -------- | --- | ----- |
| [Author et al.] [ref] | [Dataset name] |     | [Method] |     | [XX%] |
| ...                   | ...            |     | ...      |     | ...   |
Table 1. Include all 10–15 reviewed studies here, ordered chronologically or thematically.
5.3 Summary Table 2 — Methodological depth (carried into the Discussion)
This second table is reused in Section 4.4 (Discussion with Previous Works) to benchmark your own results
against the literature, so keep column definitions identical in both places.
| Title / Ref. | No. of | No. of     | Hyperparamet | Methodology | Accuracy |
| ------------ | ------ | ---------- | ------------ | ----------- | -------- |
|              | Data   | Parameters | er Tuning    |             |          |
[Silva et al.] [18] 6,080 [N/A or value] Scaling and BorutaShap, RF 86%
merging
| [...]        | [...] | [...]   | [...]    | [...]    | [...] |
| ------------ | ----- | ------- | -------- | -------- | ----- |
| Our Proposed | [N]   | [value] | [method] | [method] | [XX%] |
Study
Table 2. Bold the last row ("Our Proposed Study") so readers can immediately compare it against prior work.

6. Materials and Methods
6.1 Proposed Methodology
Open with one paragraph giving a bird's-eye view of the pipeline, then insert the system diagram, then walk
through it stage by stage.
Figure 1. Block diagram of the proposed system, showing the flow from raw data fi preprocessing fi feature
extraction/selection fi model training fi evaluation fi deployment.
Diagram description: Describe each block left-to-right or top-to-bottom in the order the data actually flows:
input data, preprocessing steps, feature engineering, the model(s), and the output/decision. Reference the
figure explicitly, e.g. "As shown in Fig. 1, the raw images are first [step]..."
Algorithm 1 (pseudocode) — template
Algorithm 1: [Name of proposed method]
Input: Dataset D, hyperparameters q
Output: Trained model M, predicted labels n
1. Preprocess D fi D¢ (cleaning, normalization, augmentation)
2. Extract/select features F from D¢
3. Split D¢ into D_train, D_val, D_test
4. Initialize model M with hyperparameters q
5. For each epoch/iteration: train M on D_train; validate on D_val
6. Select best M based on validation performance
7. Evaluate M on D_test fi n
8. Return M, n
6.2 Dataset
(cid:127) Source and provenance (public repository, hospital, self-collected) with a citation/link.
(cid:127) Total number of samples/images/records and class-wise distribution (a small table works well here).
(cid:127) Any class imbalance and how it was handled.
(cid:127) Train/validation/test split ratio.
6.3 Data Preprocessing
(cid:127) Cleaning steps: missing-value handling, noise removal, deduplication.
(cid:127) Normalization/standardization/resizing procedures.
(cid:127) Augmentation — list every transformation applied (rotation, flip, zoom, brightness, SMOTE for tabular
imbalance) with parameter ranges.
(cid:127) Feature tuning/selection — method used (e.g., BorutaShap, PCA, mutual information) and the resulting
number of retained features.
Augmentation Technique Parameter Range Applied To
Rotation ±[X]° Training set only
Horizontal/Vertical flip p = [value] Training set only
Zoom [range] Training set only
Brightness/contrast [range] Training set only

Table 3. Augmentation summary — adapt rows to the transformations you actually used.
6.4 Machine Learning / Deep Learning Models
(cid:127) List every model tested (baseline classical ML models + your proposed deep learning model), with a
one-line rationale for including each.
(cid:127) State key architectural details for DL models: number of layers, input shape, activation functions, optimizer,
loss function, learning rate, batch size, epochs, and any regularization (dropout, early stopping).
(cid:127) State hyperparameter search strategy (grid search, random search, Bayesian optimization) and the search
space.
6.5 Web Interface / Application
If a deployment component exists, briefly describe the interface (e.g., Flask/Streamlit web app), its
input/output workflow, and how it packages the trained model for end-user or clinician use. Include a
screenshot if space allows.

7. Results and Discussion
7.1 Experimental Setup
(cid:127) Hardware/software environment (GPU/CPU, RAM, framework and version, e.g., TensorFlow 2.x / PyTorch
2.x).
(cid:127) Train/validation/test split and cross-validation strategy (e.g., 5-fold CV).
(cid:127) Evaluation metrics used, and why (accuracy alone is often insufficient for imbalanced classes — justify use
of precision, recall, F1, AUC).
7.2 Confusion Matrix
Present the confusion matrix for your best-performing model, then interpret it in words: which classes are
most often confused, and a plausible reason (visual similarity, class imbalance, label noise).
Figure 2. Confusion matrix of the proposed [model name] on the test set.
7.3 Result Analysis
(cid:127) Report performance of all tested models in one comparison table (accuracy, precision, recall, F1-score,
AUC).
(cid:127) Bold the best value per column.
(cid:127) Discuss training curves (loss/accuracy vs. epoch) to show convergence and rule out over/underfitting.
(cid:127) If applicable, report inference time / model size for deployment feasibility.
Model Accuracy Precision Recall F1-score AUC
[Baseline model 1] [XX%] [XX%] [XX%] [XX%] [XX%]
[Baseline model 2] [XX%] [XX%] [XX%] [XX%] [XX%]
Proposed model [XX%] [XX%] [XX%] [XX%] [XX%]
Table 4. Overall performance comparison across all tested models.
7.4 Discussion with Previous Works
Reuse Summary Table 2 from the Literature Review (Section 5.3), now with your own result appended as the
final row. Discuss, in prose, why your result is better, comparable, or lower — referencing concrete
differences in dataset size, preprocessing, or model complexity rather than only citing the accuracy number.
Example sentence pattern: "Compared to Silva et al. [18], who reported 86% accuracy using BorutaShap-based
feature selection with Random Forest on 6,080 samples, our proposed model achieved [XX]% on a [larger/smaller]
dataset of [N] samples. This improvement can be attributed to [reason], although [limitation] may explain the
remaining gap relative to [study]."

8. Conclusion
(cid:127) Summary of findings — restate the problem, approach, and headline result in 2–3 sentences (no new
numbers, just the key ones already presented).
(cid:127) Limitations — be specific: dataset size/diversity, single-center data, class imbalance, lack of external
validation, computational cost.
(cid:127) Future work — propose concrete next steps: larger/multi-center datasets, explainability
(SHAP/Grad-CAM), lightweight deployment for mobile/edge devices, prospective clinical validation.
"In this study, we proposed [method] for [task], achieving [XX]% accuracy on [dataset]. Despite promising results, this
study is limited by [limitation]. Future work will focus on [direction 1] and [direction 2] to improve generalizability and
real-world applicability."
9. References
Use the citation style mandated by the target conference (IEEE, APA, or ACM are most common). Keep the
style consistent across all entries, and cite every reference discussed in the Literature Review.
IEEE example:
[18] A. Silva et al., "[Paper title]," [Journal/Conference name], vol. [X], no. [Y], pp. [pages], [Year].
Checklist before submission: abstract word count within limit (cid:127) all figures/tables numbered and referenced in text (cid:127) consistent citation style
(cid:127) no orphan references (cid:127) page limit met (cid:127) keywords match conference index terms.