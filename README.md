# Gibberish Detection: From Character Statistics to Transformer-Based Text Quality Assessment

A comparative study of three approaches for detecting clean, corrupted, and gibberish-like text, designed with a practical **RAG document-ingestion quality gate** in mind.

The project explores a simple question:

> Can a lightweight model built from scratch identify low-quality extracted text effectively enough to protect a downstream RAG pipeline?

Instead of starting with a large pretrained model, the project builds the solution progressively — beginning with character-level statistics, moving to classical machine learning, and finally comparing against a pretrained Transformer.

The result is a benchmark that demonstrates how different representations of text behave when the task is not semantic understanding, but **text quality and structural consistency detection**.

---

## 1. The Motivation

Modern RAG systems depend heavily on the quality of the text entering the pipeline.

A typical document workflow looks like:

```text
PDF
 │
 ▼
Parser / OCR
 │
 ▼
Extracted Text
 │
 ▼
Text Quality Check
 │
 ├── Clean
 │      │
 │      ▼
 │   Chunking
 │      │
 │      ▼
 │   Embeddings
 │      │
 │      ▼
 │   Vector Database
 │      │
 │      ▼
 │      RAG
 │
 └── Suspicious
        │
        ▼
   Retry Parser / OCR
```

A PDF parser can successfully return text while still producing:

- broken characters
- malformed words
- random symbols
- OCR substitutions
- missing spaces
- word-order corruption
- incoherent word sequences
- encoding artifacts

If such text is embedded directly, the resulting vectors can contaminate retrieval and ultimately reduce the quality of the RAG system.

This project investigates a lightweight **text-quality detection layer** that can be placed immediately after extraction.

---

# 2. What I Built

I implemented and compared three different approaches.

### Model A — From-Scratch Character N-Gram Model

A character-level statistical language model built without a pretrained NLP model.

It learns character transition patterns from each class and uses smoothed probabilities to determine which class a new text sample most closely resembles.

### Model B — Classical Machine Learning

A feature-based classifier using:

- Character TF-IDF
- Word TF-IDF
- Handcrafted text-quality features
- Logistic Regression

This provides a strong classical baseline between a pure statistical approach and a neural model.

### Model C — Pretrained Transformer

A freely available pretrained Transformer-based gibberish detector:

```text
madhurjindal/autonlp-Gibberish-Detector-492513457
```

The model is evaluated on exactly the same held-out test set as Models A and B.

This makes the comparison focused on the behavior of the three approaches under the same benchmark rather than comparing unrelated published metrics.

---

# 3. Project Evolution

The project was developed in two important versions.

## Building the Benchmark

Initial goal was to create a controlled dataset containing four types of text:

```text
clean
noise
mild_gibberish
word_salad
```

Each class contained 2,500 samples.

The version also established:

- reproducible dataset generation
- train/validation/test splitting
- three-model comparison
- common evaluation
- confusion matrices
- ROC-AUC
- PR-AUC
- MCC
- latency
- throughput
- model-size measurement
- automated report generation

The important idea in V2 was:

> All models should face exactly the same benchmark.

That makes the comparison reproducible and meaningful.

---

# 4. Making the Benchmark Rigorous

Later strengthened the dataset-generation process.

The key improvement was **global uniqueness**.

Rather than checking duplicates independently inside each class, V3 maintains a global set of previously generated samples.

Conceptually:

```python
global_seen = set()

for each class:
    generate candidate

    if candidate not in global_seen:
        add candidate
        global_seen.add(candidate)
```

This guarantees that the same text cannot accidentally appear under two different labels.

The final dataset contains:

```text
Total samples       : 10,000
Samples per class   : 2,500
Duplicate texts     : 0
Unique texts        : 10,000
```

The dataset is split using a stratified 70/15/15 split:

```text
Training     : 7,000
Validation   : 1,500
Testing      : 1,500
```

Each split preserves the class balance.

The test set therefore contains:

```text
clean             375
noise             375
mild_gibberish    375
word_salad        375
```

This creates a controlled and reproducible benchmark for all three models.

---

# 5. Dataset Design

The dataset contains four classes.

## Clean

Natural, meaningful text representing legitimate content.

The generated examples include technical, academic, financial, machine-learning, and RAG-related language.

Example structure:

```text
The retrieval system converts documents into vector representations
before storing them in a searchable vector database.
```

---

## Noise

Highly corrupted character-level content.

Examples can contain:

```text
random characters
keyboard sequences
symbols
character fragments
unstructured strings
```

The purpose is to represent severe extraction or encoding corruption.

---

## Mild Gibberish

Text that still resembles legitimate language but contains corruption.

The generator introduces transformations such as:

```text
character deletion
OCR-like substitutions
misspellings
partial word corruption
keyboard noise
```

For example:

```text
document → documnt
retrieval → retrival
algorithm → algortihm
```

This class is particularly relevant to real document-processing pipelines because corrupted OCR often remains visually or statistically similar to real language.

---

## Word Salad

Sequences composed of individually valid-looking words but lacking coherent structure.

Conceptually:

```text
retrieval algorithm document financial
embedding system probability database
signal model training vector optimization
```

The individual words can look legitimate while the overall sequence lacks meaningful structure.

This allows the benchmark to distinguish between:

```text
character corruption
```

and

```text
structural / linguistic incoherence
```

---

# 6. Model A — From-Scratch Character N-Gram

The first model intentionally avoids pretrained NLP libraries.

It learns character-level statistics directly from the training data.

For a trigram model:

```text
"model"
```

can be represented approximately as:

```text
<mo
mod
ode
del
el>
```

The model learns how frequently character sequences occur in each class.

---

## Probability Estimation

For an n-gram:

$$
P(c_i \mid c_{i-n+1}, \ldots, c_{i-1})
$$

the model estimates the probability of the next character given its preceding context.

To prevent unseen n-grams from receiving zero probability, add-$k$ smoothing is used:

$$
P(c \mid h) =
\frac{N(h,c) + \alpha}
{N(h) + \alpha V}
$$

where:

* $N(h,c)$ = count of character $c$ following history $h$
* $N(h)$ = total occurrences of history $h$
* $\alpha$ = smoothing parameter
* $V$ = vocabulary size


---

## Log-Likelihood

For a text containing characters $(c_1, \ldots, c_T)$, the model calculates the average log probability:

$$
L(x) =
\frac{1}{T}
\sum_{i=1}^{T}
\log P(c_i \mid h_i)
$$

A text that follows the character patterns learned for a particular class receives a higher score for that class.

The model calculates this score independently for all four classes.

---
## Class prediction

Let:

$$
S_k(x)
$$

be the log-likelihood score of text $x$ under class $k$.

The predicted class is:

$$
\hat{y} = \arg\max_k S_k(x)
$$

For comparison purposes, the class scores are also converted into normalized exponential scores:

$$
p_k =
\frac{e^{S_k}}
{\sum_j e^{S_j}}
$$

These values are used as normalized model scores during evaluation.


---

# 7. Why Character-Level Modeling?

The task is fundamentally different from conventional text classification.

We are not primarily asking:

> "What does this document mean?"

We are asking:

> "Does this text look structurally like valid text?"

Character patterns are extremely useful for this.

For example:

```text
document
```

contains very different character statistics from:

```text
d0cum3nt@@#
```

Even when the semantic meaning is completely lost, character-level regularities can still provide a strong signal.

This makes character models attractive for:

- OCR quality detection
- encoding corruption
- parser failures
- random character insertion
- malformed extraction
- text preprocessing pipelines

---

# 8. Model B — Classical Machine Learning

The second approach uses a conventional supervised machine-learning pipeline.

The model combines two types of TF-IDF representations.

### Character TF-IDF

Character n-grams:

```text
2–5 characters
```

This captures local spelling and character structure.

### Word TF-IDF

Word-level n-grams:

```text
unigrams + bigrams
```

This captures word-level structure.

The resulting representation is combined with handcrafted text-quality features.

---

# 9. Handcrafted Features

The classical model also uses structural features such as:

```text
text length
word count
alphabetic character ratio
digit ratio
special-character ratio
unique-character ratio
average word length
single-character word ratio
repeated-word ratio
```

For example, the alphabetic ratio can be represented as:

$$
R_{\text{alpha}} =
\frac{\text{number of alphabetic characters}}
{\text{total characters}}
$$

Similarly:

$$
R_{\text{digit}} =
\frac{\text{number of digits}}
{\text{total characters}}
$$

and:

$$
R_{\text{special}} =
\frac{\text{number of special characters}}
{\text{total characters}}
$$

These features provide explicit signals about the structural quality of the text.


---

# 10. Logistic Regression

The combined feature representation is passed to Logistic Regression.

For class \(k\):

$$
P(y=k \mid x) =
\frac{e^{w_k^T x + b_k}}
{\sum_j e^{w_j^T x + b_j}}
$$

The classifier learns a decision boundary between the four text-quality classes.

This model serves as the project's **classical ML baseline**.

---

# 11. Model C — Pretrained Transformer

The third approach uses an existing pretrained Transformer-based gibberish detector:

```text
madhurjindal/autonlp-Gibberish-Detector-492513457
```

The model provides four labels corresponding to:

```text
clean
mild gibberish
noise
word salad
```

The model contains approximately:

```text
66.96 million parameters
```

It was used as an external pretrained baseline rather than retraining it on the project's synthetic dataset.

This provides an important comparison:

```text
From scratch
      vs
Classical ML
      vs
Pretrained Transformer
```

---

# 12. Common Evaluation Protocol

A major design principle of this project is that all models are evaluated on the **same held-out test set**.

```text
                 ┌── Model A
                 │
Same Test Set ───┼── Model B
                 │
                 └── Model C
```

This prevents differences in test data from influencing the comparison.

The V3 test set contains 1,500 samples:

```text
375 clean
375 noise
375 mild_gibberish
375 word_salad
```

---

# 13. Evaluation Metrics

Multiple metrics are used because accuracy alone does not fully describe a multiclass detector.

## Accuracy

$$
\text{Accuracy} =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

Measures the overall fraction of correct predictions.

---

## Precision

$$
\text{Precision} =
\frac{TP}{TP + FP}
$$

Measures how many predicted instances of a class were actually correct.

---

## Recall

$$
\text{Recall} =
\frac{TP}{TP + FN}
$$

Measures how many actual instances of a class were detected.

---

## F1 Score

$$
F1 =
2 \frac{\text{Precision} \cdot \text{Recall}}
{\text{Precision} + \text{Recall}}
$$

Balances precision and recall.

---

## Macro F1

The F1 score is calculated independently for every class and then averaged:

$$
F1_{\text{macro}} =
\frac{1}{K}
\sum_{k=1}^{K} F1_k
$$

This gives every class equal importance.

---

## Matthews Correlation Coefficient

MCC provides a correlation-based measure of prediction quality.

For binary classification:

$$
MCC =
\frac{TP \cdot TN - FP \cdot FN}
{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

For multiclass classification, the generalized MCC is used.

MCC is useful because it remains informative when the class distribution is not perfectly balanced.

---

## ROC-AUC

ROC-AUC measures how effectively the model ranks classes across different decision thresholds.

For the multiclass experiment, one-vs-rest macro ROC-AUC is reported.

---

## PR-AUC

Precision-Recall AUC evaluates the relationship between precision and recall across thresholds.

Macro PR-AUC is reported for the multiclass task.

---

# 14. Efficiency Evaluation

Classification quality is only part of the problem.

A quality gate may be executed on thousands of document chunks, so computational efficiency matters.

Therefore the benchmark also measures:

## Latency

$$
\text{Latency} =
\frac{\text{Total inference time}}
{\text{Number of samples}}
$$

Reported in milliseconds per sample.

## Throughput

$$
\text{Throughput} =
\frac{\text{Number of samples}}
{\text{Inference time}}
$$

Reported as samples per second.


### Model Size

The serialized model size is also measured where applicable.

This allows the experiment to compare:

```text
accuracy
quality
speed
memory footprint
```

rather than looking at accuracy alone.

---

# 15. Results

The final common-test evaluation produced the following results.

| Model | Accuracy | Macro F1 | MCC | ROC-AUC | PR-AUC | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Statistical N-Gram | **86.80%** | **86.72%** | **0.826** | **0.966** | **0.890** | **0.263 ms** |
| Classical TF-IDF + LR | 49.93% | 46.20% | 0.336 | 0.571 | 0.532 | 0.315 ms |
| Transformer | 53.93% | 47.17% | 0.480 | 0.865 | 0.673 | 34.05 ms |

The statistical model also achieved:

```text
Model size       : 0.398 MB
Throughput       : 3,809 samples/sec
```

The Transformer benchmark achieved approximately:

```text
Throughput       : 29 samples/sec
```

on the evaluation environment.

---

# 16. Discussion

The most interesting outcome of the experiment is that the smallest and simplest model performed extremely well on this benchmark.

The character n-gram model achieved:

$$
\text{Accuracy} = 86.8%
$$

and:

$$
F1_{\text{macro}} = 86.72%
$$

while requiring only approximately:

$$
0.398\text{ MB}
$$

for the saved model.

The experiment demonstrates that **text-quality detection does not necessarily require a large neural architecture**.

For this particular synthetic benchmark, character-level statistical patterns provide a strong signal.

This is particularly attractive for systems where the detector needs to be:

- lightweight
- fast
- inexpensive
- easy to deploy
- independent of GPU infrastructure
- suitable for high-volume preprocessing

---

# 17. Where the Statistical Model Excels

The class-wise results show particularly strong recognition of severe corruption.

For the statistical model:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Clean | 70.40% | 83.73% | 76.49% |
| Noise | 100.00% | 98.67% | 99.33% |
| Mild Gibberish | 78.64% | 64.80% | 71.05% |
| Word Salad | 100.00% | 100.00% | 100.00% |

The primary modeling challenge is distinguishing **clean text from mildly corrupted text**.

This is also one of the most important cases for a practical document-ingestion system because severe corruption is relatively easy to identify, whereas subtle OCR/parser corruption can still look like legitimate language.

---

# 18. Confusion Matrix

The final statistical model confusion matrix is:

```text
                  Predicted

                  Clean  Noise  Mild  Salad

Actual Clean       314     0     61     0
Actual Noise         0   370      5     0
Actual Mild        132     0    243     0
Actual Salad         0     0      0   375
```

The model completely separates the word-salad class in this benchmark and almost completely separates the noise class.

The largest interaction occurs between:

```text
clean
   ↕
mild_gibberish
```

which provides a natural direction for future improvements.

---

# 19. Why the Comparison Matters

The three models represent three different philosophies.

### Statistical

```text
Learn local character patterns
          ↓
Estimate class likelihood
          ↓
Classify
```

Advantages:

- extremely lightweight
- fast inference
- no pretrained model dependency
- easy to understand
- easy to reproduce

---

### Classical ML

```text
Text
 ↓
TF-IDF
 +
Handcrafted Features
 ↓
Logistic Regression
 ↓
Prediction
```

Advantages:

- interpretable feature engineering
- established ML methodology
- relatively small model
- fast inference

---

### Transformer

```text
Text
 ↓
Tokenizer
 ↓
Transformer Encoder
 ↓
Classification Head
 ↓
Prediction
```

Advantages:

- pretrained linguistic representation
- strong contextual modeling
- reusable pretrained knowledge
- capable of learning complex text patterns

The experiment therefore becomes a study of **representation versus complexity**.

---

# 20. RAG Quality-Gate Architecture

The intended application is a quality-control layer for document ingestion.

```text
                 ┌──────────────────┐
                 │       PDF        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Parser / OCR     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Extracted Text   │
                 └────────┬─────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Gibberish / Quality   │
              │ Detector               │
              └───────────┬────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
              Clean             Suspicious
                │                   │
                ▼                   ▼
           Chunking            Retry parser/
                │              alternate OCR
                ▼                   │
           Embeddings              │
                │                   │
                ▼                   │
          Vector Database           │
                │                   │
                ▼                   │
               RAG ◄───────────────┘
```

The detector therefore acts as an **early quality-control layer**, rather than being the final objective of the RAG system.

---

# 21. Why This Design Is Useful

A quality gate can prevent poor extracted text from silently entering the retrieval pipeline.

Instead of:

```text
Bad extraction
      ↓
Chunk
      ↓
Embed
      ↓
Store
      ↓
Retrieve bad information
```

the system can perform:

```text
Bad extraction
      ↓
Detect
      ↓
Retry extraction
      ↓
Validate again
      ↓
Continue
```

This turns gibberish detection into part of a **self-healing ingestion pipeline**.


---
# 22. Limitations and Next Stage

The current benchmark is intentionally controlled and synthetic.

That makes it useful for understanding model behavior, but real document extraction introduces additional forms of corruption.

The next stage is therefore to construct a **real-world PDF/OCR benchmark** containing:

```text
clean PDF extraction
OCR corruption
missing spaces
character substitutions
encoding errors
broken reading order
repeated headers and footers
table extraction errors
equation corruption
technical documents
scientific documents
code
URLs
identifiers
```

The objective is to test whether the detector can distinguish:

```text
Unusual but valid technical text
              vs
Actually corrupted text
```

This is especially important for RAG systems because legitimate scientific and technical text can naturally contain symbols, equations, abbreviations, identifiers, and unusual terminology.
