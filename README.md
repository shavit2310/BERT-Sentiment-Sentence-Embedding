## Sentiment Sentence embedding

## Overview

Encoders failed to convey the meaning of a full sentence, although word embedding encoders have shown successes in providing good solutions for various NLP tasks. 
The difficulty of how to capture the relationships among multiple words remains a question to be solved.  
This project highlights the results achieved by SimCSE research, which improved the average of 76.3% compared to previous best results. 
SimCSE, is a simple contrastive sentence embedding framework, which can produce superior sentence embeddings, from either unlabeled or labeled data. 
The supervised model simply predicts the input sentence itself with only dropout used as noise. It’s a framework, based on the BERTbase model and evaluated with 
SentEval of FaceBook research (STS).

## Main concept
BERT
A transformer-based ML technique for (NLP) pre-training, introduced in 2019 by Google, and has become a ubiquitous baseline in NLP experiments.

The **Transformer** architecture that selectively concentrates on a discrete aspect of information, whether considered subjective or objective, encourages the option of training parallelization which led to the development of a pretrained systems like BERT and GPT. 

Created on a **pre-trained** on a large corpus of unlabelled text including the entire Wikipedia(that’s 2,500 million words!) and Book Corpus (800 million words). 

Includes a **Bidirectional** in which learns information from both, left and right, sides of a token’s context during the training phase. Which is important for truly understanding the meaning of a language. 

This characteristic and the ELMO solution for same words having different meanings based on their context (Polysemy), were the foundations for BERT implementation under 
the concept that Transfer Learning in NLP = Pre-Training first and Fine-Tuning afterwards. 

BERT input embedding combined of 3 embeddings:
**Position Embeddings**: Uses positional embeddings to express the position of words in a sentence.
**Segment Embeddings**:  Can also take sentence pairs as inputs for tasks (Question-Answering).
                         That’s why it learns a unique embedding for the first and the second sentences to help the model distinguish between them.
**Token Embeddings**:    The embeddings learned for the specific token from the WordPiece token vocabulary  Tokenization is the process of encoding a string of text into
                         transformer-readable token ID integers. From human-readable text to transformer-readable token IDs.
 
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

Torch
PyTorch version higher than `1.7.1` 

```bash
pip install torch==1.7.1
```
Scipy 

An open-source Python library, which claims to provide “industrial-strength natural language processing,”.It is the fastest-running solution. Contain pre-built models: Named entity recognition,part-of-speech (POS) tagging, and classification.Has optimized and added functions that are frequently used in NumPy and Data Science.

```bash
pip install scipy
```
Transformers

Implementation for the architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease.

```bash
pip install transformers
```

faiss

An efficient similarity search library.  The faiss is not a necessary dependency for simcse package, if error pop-up, follow the trubleshooting instractions.

## Script invocation
python3 main.py --sentences path/to/sentences.json

### Evaluation
FaceBook research (STS) evaluates sentence embeddings on semantic textual similarity tasks and downstream transfer tasks.

The FaceBook encoder based on a bi-directional LSTM architecture with max pooling, trained on the Stanford Natural Language Inference (SNLI)dataset 
delivered better encoder results than the known popular approaches like SkipThought or FastSent.

### Trained model Background 

**Data**

For supervised SimCSE, 1 million sentences from NLI datasets were sampled. 

The **sup-simcse-bert-base-uncased** model bases on the BERT uncased pre-trained model.
Meaning, the text has been lowercased before WordPiece tokenization.

## Evaluate the similarity results

Similarity score is continuous number from 0 (lowest-not related at all), to 1 (highest-equal sentences). since its calculated score, once a while it little bit higher the 1 or lower. The scores are rounded to 4 digits after the dot. The default value, as set by SimSCE is 0.6. In order to evaluate the similarities and determine on the thresholds value, the generated scores and texts reviewed. As a whole, around 80% most of the generated sentences were accurate, but there are some exceptions.
The following list defines some of characteristics, that tested specifically, individually and together (hard to validate complexity characteristics):

- Diverse meaning for a word in different location in the sentence,
- Tenses,
- Plural nouns,
- Disposal words,
- Spelling mistakes, 
- One word changes as in the subject or adjective, predict on the syntax level,
- Negativity - Positively  sentences,
- Prepositions like: till. until, on, im etc,
- Punctuations like: !, ?, "," etc,
- Street lingo (jargon),
- A definite uses like: 'A' vs 'The'
- Upper \ lower case,
- Phrases, like: give you up vs give you pencil,
- Mistakes \ errors in the similarity score .

## Result Analyzing:

First of it's important to mention that multiple sentence can be retrieved, if the THRESHOLDS boundary is met. And if the embedded input file contains the 
searched sentence more than once, it will be retrieved as the number of times it appears.
Important to emphasize that when a sentence generates a similar sentence, it will get the same score when it is produced in reverse order.
Not surprisingly, the result shows no distinguish between sentences that contain words with uppercase / lowercase, since all texts lowercase, before WordPiece tokenization.

Following to the list above, here some examples:

On Tenses aspect: "The boy jump to the pool" vs "The boy jumps to the pool" scored 0.9735, but “He is gone” vs ”he left”, drops the score, mistakenly,to 0.7627.
while “She was hungry” vs She is hungry" scored 0.8814And it decrease slightly when compering to “She has been hungry“ (0.8674) but surprisingly increases for the 
following mistake: “She have been hungry” (0.9079).
When changing gender to "he" - “He is hungry”, the score drops to 0.7932O.
Continue to on plural similarity check, and "He is sleeping" vs "We are sleeping" dropped to 0.6564. 
Seems that plural\single issue as the gender aspect is significant.

When it comes to spelling mistakes, it seems no gambles are taken: it constantly drops the score, like in:"This cource is good" vs "This course is good" scored 0.4905. but  
on a missing meaning mistake:"We will stay tii you go" vs "We will stay only for you", scored is up to 0.7471, Which is very close score to an accurate similarity, as:
"We will stay only for you" vs "We will stay until you come" scored on 0.8.
Maybe unclear word 'weight' less, and a similarity sentence can be reached.

On the negativity aspect “The clerk is not nice” scored less than 0.4 similarity to the positive “The clerk is nice”, and rightly to “The clerk is rude” as 0.8976. 
But surprisingly (and wrongly) vs to “The nice clerk left” scored with 0.5621. 

A strongly, influenced detected for punctuations ;
Too low score for "A man is playing a guitar" vs "Who is playing the guitar?" (0.6753).
"Never!! gonna give you up?" vs "Never gonna give you up" (0.6989).

And when it comes to Street lingo (jargon), like “gonna” and “wonna";
"I am going to stop talking with you” vs “I gonna stop talking with you”,
the accurate score is 0.9470. 

A definite uses like: 'A' vs 'The’, seems to have minor influence:
“The cellulars is not stolen”, vs “And, a cellulars is not stolen” (0.9708), 
which reveals that unrelated words (meaning aspects),does not have any effect. 

In other example, where the unfinished sentence might have some significance, the score dropped to 0.9010: 
 “a cellulars is not stolen, back to”, ivs “The cellulars is not stolen" (0.9010). 

When it comes to adjectives in the sentence, 
various scorers attached:“The ugly cellular is stolen” vs “The pretty cellular is stolen” (0.8650).
Founded to be slightly Below , for a adjective omission: 
“The pretty cellular is stolen” vs “The cellulars is stolen” (0.8791),and surprisingly, 
on a mistake: ”The pretty cellular is stolen” vs “The pritty cellular is stolen” increases to 0.8832.
On the sentence's subject it clearer: “The cat is running” vs “The dog is running” scored very low, limit of 0.4155 And on the predict (in grammar): 
“the dog is hungry” vs “The dog is running” scored little bit higher, but not much (0.5790).
 
On prepositioning, it depends, if it's a phrase:
"I will give you up” vs “I will give you a pencil” scored under 0.4, And on 
till\until wrongly similarity score is too high - 0.9012, thus it's  opposite meaning.
"We will go until you come" vs "We will stay until you come".

On the diverse meaning for a word in different location in the sentence.
The output file contain several examples:
“The drill is missing” vs “I have been to the dentist many times, so I know the drill” , rightly, was not founded, matched. 
And scored low grade: “She had a boyfriend with a wooden leg but broke it off” vs “she broke the partnership”(0.5384)
“A fight broke out between the stepmother and the man before her death” (0.4313)
where as “she broke the partnership” vs “she broke the table” scored as 0.5220, which can have been considered as similar meaning, but then relatively to “She left” that scored 0.5194 (which should'nt be found matches, at all).
"life is pointless” vs “Without geometry life is pointless” (wrongly, 0.6498).
Where interest in full meaning was not achieved, when below 0.4 to “The subject aroused interest” vs “I used to be a banker, but comparing to "I lost interest” or to “As an inverter, I lost interest”, it should have been higher.

## Conclusion

There is still a long way to go with fine-tuning toAchieve higher accuracy. There are sentences that have a question mark about them. 
We are inspired to handle a large number of sentences on the one hand, and on the other hand maintain a high level of accuracy and reliability. 
That is why the value of the thresholds is important. Full confidence is to set the thresholds  to 0.8000. 
A minor number of sentences will be generated incorrectly. But, after reviewing the data set, I decided to go lower, setting the threshold to 7.620. 
This may result in more accidentally created sentences, but mass generated sentences required, also.

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

3. On RuntimeError: Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.
   Be aware: The search results reliability decreases on brute force search
   
   
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

## Citation

Credit for the SimCSE project:

```bibtex
@inproceedings{gao2021simcse, title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
 author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi}, booktitle={Empirical Methods in Natural Language Processing (EMNLP)}, year={2021}
}
```
## References

Details on the [SimSCE](https://arxiv.org/pdf/2104.08821.pdf) framework research.<b/>
Details on [SentEval](https://research.fb.com/downloads/senteval/) the evaluation code for sentence embeddings. <b/>
Introduction to the World BERT [BERT](https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/). <b/>
[Bert_uncased](https://huggingface.co/bert-base-uncased) pre trained models. <b/>
