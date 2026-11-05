<!--
Editable source for Week5_Homework_Solutions.pdf.
Build from the course workspace:
  .venv/bin/python 'AI Strategy Assignment/Week5_Solutions.py' --pdf-only
Dependencies for PDF only: reportlab, markdown-it-py.
Page breaks, paragraph styles and table widths are print-layout hints.
-->

# Week 5 | In-class challenge

<!-- style: SubX -->

Proposed solution prepared from the prompt; no completed team response was supplied. [1, 2]

<!-- widths: 113, 112, 279 -->

| Need | Technique | Reason |
| --- | --- | --- |
| a. Logbook shorthand | CPT, if needed | CPT learns domain phrasing; first test a glossary plus a pretrained model. |
| b. Current APU MEL + citation | RAG | Retrieve the approved revision filtered by fleet and aircraft applicability, and cite it. |
| c. Route defect to ATA chapter | CFT | A supervised classification head maps the write-up to a fixed ATA label set. |
| d. Standard shift summary | Prompting, then IFT | Try a template with examples; use IFT only if format or coverage remains inconsistent. |
| e. Tire replacements last quarter | NL-to-SQL | Query dated fleet records with validated definitions; document retrieval cannot reliably count events. |
| f. No sign-off; uncertainty | DPO + guardrails | Expert preferences teach deferral; explicit application rules block sign-off claims and unsupported answers. |

## 2. Sizing the project

<!-- style: SmallX -->

Assume **decimal GB** and the lecture's averages: 10 x 10<sup>9</sup> / 4 = **2.5B tokens**. Scans, images, tables, duplicates and boilerplate reduce usable text; measure tokens after OCR, cleaning and deduplication. A 3B model wants roughly 20 x 3B = **60B tokens** under Chinchilla: this corpus is **24x too small** (4.17% of the rule-of-thumb target). Scratch training is technically possible, but is a poor data-efficient choice; repeating the same corpus does not add that much distinct information.

<!-- style: SmallX -->

One 8B CPT epoch: C = 6 x 8 x 10<sup>9</sup> x 2.5 x 10<sup>9</sup> = **1.2 x 10<sup>20</sup> FLOPs**. At 4 x 10<sup>14</sup> FLOP/s: 300,000 GPU-seconds = **83.3 H100 GPU-hours**. This is aggregate compute, before overhead or extra data; mixed-precision Adam also needs about **128 GB + activations**, so sharding may be needed. Real cost: clean, permitted data, expert labels/evaluation, integration and revision control as well as compute.

## 3. Evaluate before training

<!-- style: SmallX -->

Maintenance engineers, technicians and dispatch SMEs write a held-out golden set with source revisions, correct answers and rubric scores. Include shorthand, aircraft applicability, updated/superseded procedures, unanswerable questions, ATA routing, summaries and SQL counts. Split by time and aircraft/work-order groups; keep related records and paraphrases out of training. Baselines: **(1)** general model + prompt/glossary, **(2)** that model + hybrid RAG, **(3)** small open model + the same RAG. Score evidence-supported correctness, citation validity, harmful errors, abstention, routing macro-F1, summary fidelity, exact counts and latency. Stop without training if an untrained baseline meets agreed quality, reliability and deployment targets and training offers no material held-out gain.

## 4. Risk and recommendation

<!-- style: SmallX -->

**Largest weights risk:** confidently recalling a superseded procedure without traceable revision evidence. Keep authoritative procedures in versioned retrieval; validate applicability/citations and abstain on gaps. **Data risk:** OCR can corrupt part numbers, units or limits; review critical tables against approved originals. **Recommendation to the VP:** start with evaluated hybrid RAG and database queries, add task-specific CFT/IFT where needed, and use CPT only for a demonstrated language gap.

<!-- pagebreak -->

# Question 1 | Combine the features

<!-- style: SubX -->

Recommendation: a small two-branch classifier, tested against a strong 770-dimensional baseline.

Let **e** be the supplied 768-dimensional text embedding and **s** contain amount and account age. Assume all features are available at transaction decision time and the target is a binary fraud label. Freeze the text encoder initially so the comparison tests the fusion design.

```text
A. EARLY CONCATENATION

Text embedding (768) -> LayerNorm ---------+
                                         +-> Concatenate (770)
Amount + age (2) -> log1p + train scaling -+          |
                                                    v
                                         MLP (64) -> logit -> sigmoid

B. LINEAR FUSION

Text (768) -> LayerNorm -> Linear (64) -> ReLU --+
                                              +-> Concatenate (80)
Scaled amount + age (2) -> Linear (16) -> ReLU -+          |
                                                         v
                                           Linear (32) -> ReLU
                                                         |
                                                         v
                                                  logit -> sigmoid
```

## A. Concatenate first

Yes: concatenate the two variables to obtain **770 features**, then train logistic regression or an MLP. A learned layer has a separate coefficient for each input; an MLP can also learn interactions between text and numbers. This is a reasonable, inexpensive baseline. A linear model alone cannot learn nonlinear interactions without explicit feature engineering. We might run into issues with important variables getting lost in the mud due to the size of the feature space. 

## B. Process separately, then fuse

Use LayerNorm(e) -> Linear(768, 64) -> ReLU for text, and scaled s -> Linear(2, 16) -> ReLU for structured features. Concatenate the resulting **80-dimensional representation**, then use Linear(80, 32) -> ReLU -> Linear(32, 1). Train both branches and the fusion head jointly with binary cross-entropy on logits; apply sigmoid for inference. Separate branches permit modality-specific preprocessing, capacity and regularization, while the fusion MLP learns cross-modal interactions without losing feature importance.

## Transform the two variables before either architecture

For nonnegative, right-skewed amount and age, try **log1p**, then standardize each feature using training-set means and standard deviations. Compare this with ordinary or robust scaling on validation data; log transforms are a defensible starting choice, not a requirement. If refunds can be negative, use a signed-log transform or a separate refund flag. Validate units and ranges, impute missing values using training-set statistics, and consider missingness flags. Two extra flags would make the simple input **772** rather than 770 dimensions.

Keep preprocessing fixed at inference. Normalize the text representation separately if its scale warrants it; LayerNorm in the accompanying example is one option. Standardizing numbers controls scale, but does not force equal influence across modalities.

<!-- style: TinyX -->

Sources: homework Q1 [3]; trainable projections and concatenation in the supplied attention notebooks [4, 5].

<!-- pagebreak -->

# Question 1 | Influence and validation

<!-- style: SubX -->

Feature count is not the same thing as predictive importance.

For a linear fusion model, write the logit as:

```equation
z = w_text^T e + w_amount s_amount + w_age s_age + b
```

The two numeric coefficients can be large while most text coefficients are small or zero. For example, set all 768 text coefficients to zero and use **z = 3s_amount - 2s_age**. Holding age and text fixed, changing standardized amount from 0 to 1 changes sigmoid(z) from **0.500 to 0.953**. The two structured variables determine the decision despite having fewer dimensions. This is a constructive example, not a claim about real fraud data.

Dimension imbalance can still affect optimization: a large text block may have a larger total norm, many correlated features, or more opportunities to fit noise. Scaling, training signal, regularization, initialization and data quality matter. Expanding two numbers into 16 hidden units does not create new information or guarantee their importance.

<!-- widths: 111, 198, 195 -->

| Architecture | Benefit | Limitation |
| --- | --- | --- |
| Early concatenation | Simple; few design choices; useful low-variance baseline. | One shared stack must handle both modalities; poorly scaled inputs can impair training. |
| Two branches + fusion | Separate transforms and capacity; joint nonlinear interactions. | More tuning; bottlenecks may discard signal; extra capacity can overfit. |

## Recommendation and decision rule

I recommend **two branches with a modest fusion head** because the features differ in scale and structure, and amount/age may interact with textual patterns. Keep early concatenation as the control and select it instead if held-out results are equivalent. Match tuning effort and regularization; do not infer superiority from the diagram.

Use chronological validation and prevent account or linked-transaction leakage as appropriate to deployment. Fit all preprocessing on training data only. Compare text-only, structured-only, early fusion and two-branch models. For imbalanced fraud, assess precision-recall AUC, recall at the permitted false-positive rate, calibration and cost-based thresholds. Use feature ablations or permutation tests, interpreted carefully when inputs are correlated, to check incremental contribution.

## Accompanying Python and references

<!-- style: SmallX -->

**Week5_Solutions.py** reproduces the challenge arithmetic and storage estimates, implements both fusion models, checks finite gradients, demonstrates the numerical counterexample above, and trains a tiny attention module without changing its size. These are synthetic checks; no fraud dataset or real maintenance corpus was supplied. Run the file for printed results; use **--pdf-only** to regenerate this document from **Week5_Homework_Solutions.md**. Dependencies: PyTorch for demonstrations; ReportLab and markdown-it-py for PDF generation.

<!-- style: TinyX -->

**[1]** Week5_Challenge_Airline_Domain_Model.html, Tasks 1-4.  
**[2]** Week5_Lecture_Note_Domain_Models.html, sections 2 and 4-8.  
**[3]** Week5_Homework.html, Questions 1-3 and submission requirements.  
**[4]** Karpathy_GPT_Teaching_Edition.ipynb, sections 4-8; raw cells 20-22 and 48-50.  
**[5]** Self_Attention_Sebastian_CH3.ipynb, sections 3.4-3.6; raw cells 24, 68, 87, 90 and 128.  
All references are the supplied course files. Hardware/storage figures are estimates under stated assumptions, not device benchmarks.

<!-- pagebreak -->

# Question 2 | What learning changes

<!-- style: SubX -->

**No. Learning a new skill does not necessarily make a model larger.** [2, 4, 5]

Training can change the values of existing parameters while keeping the architecture and parameter count unchanged. Adding a new component introduces new parameters and may increase the model's size.

## A. What is a model parameter, and what happens to its value during training?

A model parameter is a learned numerical value, such as an entry in a weight matrix, a bias, or an embedding table. These values determine how the model converts inputs into predictions.

During training, the model computes a loss, backpropagation calculates gradients, and an optimizer updates the trainable parameter values to reduce that loss. **Changing these values does not require adding parameters.**

## B. Can the same architecture behave differently after training without changing its parameter count?

**Yes.** The same architecture can produce different outputs because its existing weights have changed. For example, updating attention projection matrices changes which input information the model emphasizes, even when the matrices retain exactly the same dimensions and number of parameters.

## C. How do techniques discussed in class support this explanation?

<!-- widths: 58, 269, 177 -->

| Technique | What changes during training? | Must the model become larger? |
| --- | --- | --- |
| **CPT** | Existing language-model weights are updated using domain text, helping the model learn specialized terminology and patterns. | **No.** The architecture can remain unchanged. |
| **IFT** | Existing weights are updated using instruction-response examples, helping the model follow instructions and produce desired formats. | **No.** New behavior can come from changing existing values. |
| **DPO** | Existing policy weights are updated using preferred and rejected responses. | **No.** A reference model used during training does not have to become part of the deployed model. |

## D. When might adaptation introduce an additional set of parameters?

**Classification Fine-Tuning (CFT)** can introduce a new classification head. For example, routing maintenance reports to ATA chapters requires a head that maps a hidden representation of size **d** to **K** class scores.

A linear head with a bias introduces **dK + K parameters**. If it is added while retaining all existing components, the total parameter count increases. If it replaces the original vocabulary output head, it introduces a new parameter set, but the total model size may decrease because the classification head can be smaller.

<!-- pagebreak -->

# Question 3 | Offline architecture

<!-- style: SubX -->

One diagram: central preparation is allowed; every inference step runs on the iPad. [2, 3]

```text
CENTRAL PREPARATION (BEFORE DEPLOYMENT)

Approved 10 GB corpus
  | OCR, tables, revision and aircraft applicability
  v
Prepare retrieval                    Prepare local models
  | Chunk + BM25 + embeddings          | Small generator + encoder
  | Source-page links                  | Optional IFT + 4-bit export
  +------------------+-----------------+
                     v
         Validated, signed versioned package
         Models + manuals + index + expiry manifest
                     |
         Install while connected
---------------------+------------------------------------------
ON THE IPAD: ALL INFERENCE IS OFFLINE
                     v
Technician question: aircraft/tail, task, symptoms
                     |
                     v
Applicability + validity checks
  | Package expiry; fleet, tail and revision filters
  v
Local hybrid retrieval <-------- Local retrieval store
  | BM25 + small query encoder    Chunks, metadata, index,
  | Select relevant evidence      approved original pages
  v                                        |
Small local generator                      |
  | Question + evidence -> draft           |
  v                                        |
Evidence checks                            |
  | Verify source IDs and support          |
  | Conflicts or gaps -> abstain            |
  v                                        v
Answer + local source viewer (document, revision, page/task links)

Expired or unsupported -> withhold current procedural advice;
refer to Maintenance Control.
```

<!-- style: SmallX -->

Weights supply language and task behavior. The installed retrieval package supplies the authoritative, versioned maintenance evidence. The following page addresses the four requested design questions.

<!-- pagebreak -->

# Question 3 | Design justification

<!-- style: SubX -->

One-page justification - four requested decisions. [2, 3]

## A. What models would run on the iPad, and how would they fit its memory and compute limits?

Run an instruction-tuned **small language model, initially about 1B parameters**, plus a compact text embedding encoder and local BM25 search. Quantize generator weights to 4-bit, use an iPad-compatible accelerated runtime, generate one answer at a time, and cap context length and output length to bound KV-cache and activation memory. A 1B model has roughly **0.5 GB of packed weights**; 3B would be about 1.5 GB, before quantization metadata, cache, encoder and runtime overhead. Move toward 3B only if measured accuracy, peak RAM, latency and thermal behavior pass on the actual target iPad. A cached glossary and direct source viewer support abbreviation handling and fallback.

## B. Where would the knowledge from the 10 GB corpus live?

Keep the 10 GB corpus **primarily in local storage**: approved originals, extracted procedure chunks, revision/applicability metadata, and a hybrid retrieval index. Weights provide general language ability and optional task training, rather than serving as the source of current procedures. Disk capacity and RAM are different constraints: keep documents and indexes on disk and load bounded search results. As an upper text scenario, 2.5B tokens in 500-token windows with 100-token overlap yield about 6.25M chunks; 384-dimensional FP16 vectors alone need **4.8 GB**, before search-index overhead. Measure extraction yield, compress or memory-map the index, and reserve storage for originals plus updates. Do not assume the entire store fits in RAM.

## C. Which techniques would you use, and what role would each play?

Use **local RAG** for current, citable facts; keyword search preserves exact part numbers and ATA codes, while a local encoder retrieves semantic matches. Filter by aircraft configuration, task and approved revision, then feed a few complete, relevant procedure passages to the small generator. The answer must link to local document pages and task IDs. Start with prompting; add **IFT** centrally on expert-verified, retrieval-grounded examples if the small model needs help with citations, format or abstention. **Quantization** reduces inference memory. Reserve **CPT** for demonstrated shorthand errors after glossary/retrieval improvements; it is not the update channel for procedures. **CFT** is unnecessary for this initial question-answering workflow; a routing head is optional if later evaluation justifies one.

## D. How would you keep the system reliable as manuals and procedures change?

Centrally ingest approved revisions, verify OCR/tables and aircraft applicability, rebuild affected chunks/index entries, and remove superseded entries from active search. Ship a signed package containing matching documents, index, models and a version/expiry manifest; install it atomically after integrity checks when connected, with a validated rollback copy. Test retrieval recall, citation support, revised facts, conflicting sources and unanswerable cases with maintenance SMEs after every package or model change. At inference, validate cited IDs and evidence support; use direct excerpts or abstain when support is weak. Display revision dates and offline freshness. An offline device cannot discover unseen revisions: after package expiry or a known applicability conflict, withhold current procedural advice and direct the technician to Maintenance Control. The assistant supplies cited support; it cannot authorize dispatch or sign off maintenance.
