## Sentiment Sentence embedding

## Overview

Encoders failed to convey the meaning of a full sentence, although word embedding encoders have shown successes in providing <br/>
good solutions for various NLP tasks. The difficulty of how to capture the relationships among multiple words remains a <br/>
question to be solved. This project highlights the results achieved by SimCSE research, which improved the average <br/>
of 76.3% compared to previous best results. <br/>
SimCSE, is a simple contrastive sentence embedding framework, which can produce superior sentence embeddings, from either <br/>
unlabeled or labeled data. <br/>
The supervised model simply predicts the input sentence itself with only dropout used as noise. It’s a framework, based <br/>
on the BERTbase model and evaluated with SentEval of FaceBook research (STS).<br/>

## Main concept

BERT<br/>
A transformer-based ML technique for (NLP) pre-training, introduced in 2019 by Google, and has become a ubiquitous baseline <br/>
in NLP experiments.

The **Transformer** architecture that selectively concentrates on a discrete aspect of information, whether considered <br/>
subjective or objective, encourages the option of training parallelization which led to the development of pretrained <br/>
systems like BERT and GPT. 

Created on a **pre-trained** on a large corpus of unlabelled text including the entire Wikipedia (that’s 2,500 million words) <br/>
and Book Corpus (800 million words). 

Includes a **Bidirectional** in which learns information from both, left and right, sides of a token’s context during the <br/>
training phase. Which is important for truly understanding the meaning of a language. 

This characteristic and the ELMO solution for same words having different meanings based on their context (Polysemy), <br/>
were the foundations for BERT implementation under the concept that Transfer Learning in NLP = Pre-Training first <br/>
and Fine-Tuning afterwards. 

BERT input embedding combined of 3 embeddings:<br/>
**Position Embeddings**: Uses positional embeddings to express the position of words in a sentence.<br/>
**Segment Embeddings**:  Can also take sentence pairs as inputs for tasks (Question-Answering). That’s why it learns a unique <br/>
                         embedding for the first and the second sentences to help the model distinguish between them.<br/>
**Token Embeddings**:    The embeddings learned for the specific token from the WordPiece token vocabulary Tokenization is <br/>
                         the process of encoding a string of text into transformer-readable token ID integers. <br/>
                         From human-readable text to transformer-readable token IDs.<br/>
 
## Getting Started

To use the demo, first install the `simcse` package 

```bash
git clone https://github.com/princeton-nlp/SimCSE
cd SimCSE
python setup.py develop
```
After installing the package, load the chosen model by the following lines of code:
(If not, follow the troubleshooting section)

```bash
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
```
Then run the following script to install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

Install the enviroment

Pull the project, and create 2 directroies: **data** (contains the input json files) and **output**, <br/>
which contains the standard output csv files. The results will be displayed on the screen, also.<br/>

Torch
PyTorch version higher than `1.7.1` 

```bash
pip install torch==1.7.1
```
Scipy 

An open-source Python library, which claims to provide “industrial-strength natural language processing”. It is the <br/>
fastest-running solution. Contain pre-built models: Named entity recognition, part-of-speech (POS) tagging, and <br/>
classification. Has optimized and added functions that are frequently used in NumPy and Data Science.<br/>

```bash
pip install scipy
```
Transformers

Architecture implementation that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease.

```bash
pip install transformers
```

Faiss

An efficient similarity search library. <br/>
The faiss is not a necessary dependency for simcse package, if error pop-up, follow the troubleshooting instructions.<br/>

## Script invocation

python3 main.py --sentences <path/to/sentences.json>

## Background of the Pretrained model  

### Evaluation

FaceBook research (STS) evaluates sentence embeddings on semantic textual similarity tasks and downstream transfer tasks.
The FaceBook encoder based on a bi-directional LSTM architecture with max pooling, trained on the Stanford Natural <br/>
Language Inference (SNLI)dataset delivered better encoder results than the known popular approaches like SkipThought or FastSent.

### Trained model 

**Data**

For supervised SimCSE, 1 million sentences from NLI datasets were sampled. <br/>
The **sup-simcse-bert-base-uncased** model bases on the BERT uncased pre-trained model.<br/>
Meaning, the text has been lowercased before WordPiece tokenization.<br/>

## Analyze results

### Evaluation scaling

Similarity score is a continuous number from 0 (lowest-not related at all), to 1 (highest-equal sentences). <br/>
since its calculated score, once a while it is a little bit higher or lower 1. The scores are rounded to 4 digits after the dot.<br/>
The default value, set by SimSCE to 0.6. Upon SimSCE threshold is required, which divides the results to Pass (1) and Fail (0).<br/>
As a whole, around 80% most of the generated sentences are accurate, but there are exceptions. <br/>
The following list defines characteristics, that review specifically, as individually and together <br/>
(hard to validate complexity characteristics):<br/>

- Diverse meaning for a word in different location in the sentence,
- Tenses,
- Plural nouns,
- Disposal words,
- Spelling mistakes,
- One word changes as in the subject or adjective, predict on the syntax level,
- Negativity - Positive sentences
- Prepositions like: till. until, on, in etc,
- Punctuations like: !, ?, "," etc,
- Street lingo (jargon),
- A definite uses like: 'A' vs 'The'
- Upper \ lower case,
- Phrases, like: give you up vs give you pencil,
- Mistakes \ errors in the similarity score.

## Review results

First, it is important to mention that multiple sentences can be generated, if the threshold boundary is met. And if the embedded input file<br/>
contains the searched sentence more than once, it will be retrieved as the number of times it appears.<br/>
Let's emphasize that when a sentence generates a similar sentence, it will get the same score when it is produced in reverse order.<br/>
Not surprisingly, there is no distinction between sentences that contain words with uppercase / lowercase, since the model training proces:<br/>
all texts lowercase, before WordPiece tokenization.<br/>

The quality of the result can be emphasized in the following example:<br/>

On Tenses aspect: "The boy** jump** to the pool" vs "The boy **jumps** to the pool" scored 0.9735,<br/>
but “He is** gone**” vs ”he **left**”, drops the score, mistakenly, to 0.7627. While “She **was** hungry” vs She **is** hungry"<br/>
scored 0.8814, and it decreases slightly, when comparing to “She **has been** hungry“ (0.8674), but surprisingly increases for the<br/>
following mistake: “**She have** been hungry” (0.9079).<br/>
When changing gender, “**He** is hungry”, the score drops to 0.7932O.<br/>
Continue on the plural similarity check, and "**He** is sleeping" vs "**We** are sleeping" dropped to 0.6564.<br/>
Seems that plural\single issue as the gender aspect is significant.<br/>

When it comes to spelling mistakes, no guesses are taken: it constantly drops the score.<br/>
like in: "This **cource** is good" vs "This **course** is good" scored 0.4905. But on a meaning missing even a mistake can increases:<br/>
"We will stay **tii** you go" vs "We will stay only for you", scored 0.7471, Which is very close score to an accurate similarity, as:<br/>
"We will stay only for you" vs "We will stay until you come" scored on **0.8**.<br/>
Maybe unclear words 'weights' less, and a similarity sentence can be reached.<br/>

On the negativity aspect “The clerk **is not nice**” scored less than 0.6 similarity to the positive “The clerk **is nice**”,<br/>
and rightly to “The clerk **is rude**” as 0.8976. But, surprisingly (and wrongly) vs “The **nice clerk left**” scored 0.5621<br/>

It is clear that punctuation has a strong influence: Too low score for: "A man is playing a guitar" vs "Who is playing the guitar?"<br/>
"Never**!!** gonna give you up?" vs "Never gonna give you up" (0.6989)<br/>

And if we are on the Street lingo (jargon), like “gonna” and “wonna";<br/>
"I am going to stop talking with you” vs “I gonna stop talking with you”, an accurate score give -  0.9470.<br/>

A definite use like: 'A' vs 'The’, seems to have minor influence:<br/>
“The cellulars is not stolen”, vs “And, a cellulars is not stolen” (0.9708), which reveals that unrelated words (meaning aspects),<br/>
does not have any effect. Here an **And** is the sentence prefix, but when the unfinished sentence, might have significance:<br/>
 “a cellulars is not stolen, **back to**” vs “The cellulars is not stolen" (0.9010).<br/>

On adjectives various scorers attached: “The ugly cellular is stolen” vs “The pretty cellular is stolen” (0.8650).<br/>
Founded to be slightly Below, for a adjective omission: “The pretty cellular is stolen” vs “The cellulars is stolen” (0.8791),<br/>
and surprisingly, on a mistake: ”The pretty cellular is stolen” vs “The **pritty** cellular is stolen” increases to 0.8832.<br/>
On the sentence's subject it clearer: “The **cat** is running” vs “The** dog** is running” scored very low, limit of 0.4155,<br/>
and on the predict (in grammar): “the dog is hungry” vs “The dog is running” scored little bit higher, but not much (0.5790).<br/>
 
On prepositioning, it depends, if it's a phrase:<br/>
"I will **give you up**” vs “I will **give you a** pencil” scored under 0.6, And on till\until wrongly similarity score<br/>
is too high - 0.9012, thus it's opposite meaning. "We will **go until** you come" vs "We will **stay until** you come".<br/>

Challenging is a diverse meaning for a word in different locations in the sentence:<br/>
“The **drill** is missing” vs “I have been to the dentist many times, so I know the **drill**”, rightly, match not found.<br/>
And, scored low on similarity check: “She had a boyfriend with a wooden leg but **broke it off**” vs “she **broke the partnership**” (0.5384),<br/>
“She had a boyfriend with a wooden leg but **broke it off**” vs “A fight broke out between the stepmother and the man before her death” (0.4313).<br/>
Where as “she **broke the partnership**” vs “she “She had a boyfriend with a wooden leg but **broke it off**” vs” scored as 0.5220,<br/>
which can have been considered as similar meaning, if the following relatively to “She left” wouldn't receive the 0.5194 score<br/>
(which should not be found matches, at all).<br/>
"life is pointless” vs “Without geometry life is pointless” (wrongly, 0.6498). Where **interest** meaning was not achieved,<br/>
on a below 0.6 to “The subject **aroused interest**” vs “I used to be a banker, but comparing to "I **lost interest**” or to<br/>
“As an inverter, I lost interest”, it should have been higher.<br/>

### Conclusion

There is, still, a long way to go on fine-tuning in order to achieve higher accuracy.<br/>
There are, still, question marks on some generated sentences. The value of the thresholds is important, because <br/>
We are inspired to handle a large number of sentences on the one hand, and on the other hand maintain a high level of <br/>
ccuracy and reliability. Thresholds could have been set to 0.8000, with a minor number of sentences generated incorrectly.<br/>
But, after careful thoughts, as the best balance, I decided to go set it lower, to 0.7620. <br/>
This may result in more accidentally generated sentences, but archives mass generated sentences.<br/>
So, if a similarity score between the input sentence and the generated one is 0.7620 or higher it rated as Pass (1), 
otherwise it rated as 0 (Fail).

## Troubleshoot

1. On RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
   Resolved by a compromise solution, replacing the library code as followed:
    ```bibtex
   On local_path..->\anaconda3\envs\project_name\Lib\site-packages\torch\tensor.py
   replace the row, as following:

   def __array__(self, dtype=None):
      if dtype is None:
        return self.numpy() -> return self.detach().numpy()
      else:
        return self.numpy().astype(dtype, copy=False)
    ```
   
2. On ModuleNotFoundError: No module named 'simcse', install:

   ```bash
   pip install SimCSE
   ```

3. On RuntimeError: Fail to import faiss. If you want to use faiss, install faiss through PyPI.<br/>
   Now the program continues with brute force search.<br/>
   Be aware: The search results reliability decreases on brute force search<br/>
   
   
   WARNING: found out that faiss did not well support Nvidia AMPERE GPUs (3090 and A100).
            In that case, you should change to other GPUs or install the CPU version of faiss package.

   For CPU-version faiss, run
 
   ```bash
   pip install faiss-cpu
   ```

   For CPU-version faiss, run

   ```bash
   pip install faiss-gpu
   ```

## References

Details on the [SimSCE](https://arxiv.org/pdf/2104.08821.pdf) framework research.<br/>
Details on [SentEval](https://research.fb.com/downloads/senteval/) the evaluation code for sentence embeddings. <br/>
Introduction to the World BERT [BERT](https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/). <br/>
[Bert_uncased](https://huggingface.co/bert-base-uncased) pre-trained models. <br/>

## Citation

Credit for the SimCSE project:

```bibtex
@inproceedings{gao2021simcse, title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings}, <br/>
author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi}, booktitle={Empirical Methods in Natural Language <br/>
Processing (EMNLP)}, year={2021}} <br/>
```
