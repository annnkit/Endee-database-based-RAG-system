"""
load_sample_data.py
Downloads a few free research abstracts so you can test the system
without needing your own PDFs.
"""

from pathlib import Path

SAMPLES = [
    {
        "title": "attention_is_all_you_need",
        "text": """
Attention Is All You Need

Abstract:
The dominant sequence transduction models are based on complex recurrent or convolutional neural
networks that include an encoder and a decoder. The best performing models also connect the encoder
and decoder through an attention mechanism. We propose a new simple network architecture, the
Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be superior in quality
while being more parallelizable and requiring significantly less time to train.

Introduction:
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular,
have been firmly established as state of the art approaches in sequence modeling and transduction
problems such as language modeling and machine translation. Numerous efforts have since continued
to push the boundaries of recurrent language models and encoder-decoder architectures.

The Transformer follows an encoder-decoder structure using stacked self-attention and
point-wise, fully connected layers for both the encoder and decoder. The encoder maps an input
sequence of symbol representations to a sequence of continuous representations. Given this
continuous representation, the decoder then generates an output sequence of symbols one element
at a time.

Multi-Head Attention:
Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O where each head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Positional Encoding:
Since our model contains no recurrence and no convolution, in order for the model to make use of
the order of the sequence, we must inject some information about the relative or absolute position
of the tokens in the sequence. We add positional encodings to the input embeddings at the bottoms
of the encoder and decoder stacks.

Results:
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the
best previously reported models including ensembles by more than 2.0 BLEU, establishing a new
state-of-the-art BLEU score of 28.4.
""",
    },
    {
        "title": "bert_paper",
        "text": """
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Abstract:
We introduce a new language representation model called BERT, which stands for Bidirectional
Encoder Representations from Transformers. Unlike recent language representation models, BERT
is designed to pre-train deep bidirectional representations from unlabeled text by jointly
conditioning on both left and right context in all layers. As a result, the pre-trained BERT
model can be fine-tuned with just one additional output layer to create state-of-the-art models
for a wide range of tasks.

Pre-training BERT:
We pre-train BERT using two unsupervised tasks. The first is Masked Language Model (MLM). In
order to train a deep bidirectional representation, we simply mask some percentage of the input
tokens at random, and then predict those masked tokens. We refer to this procedure as a masked LM.
The second task is Next Sentence Prediction (NSP). Many important downstream tasks such as Question
Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship
between two sentences, which is not directly captured by language modeling.

Fine-tuning BERT:
Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT
to model many downstream tasks—whether they involve single text or text pairs—by swapping out the
appropriate inputs and outputs.

Results:
BERT obtains new state-of-the-art results on eleven natural language processing tasks, including
pushing the GLUE score to 80.5%, MultiNLI accuracy to 86.7%, SQuAD v1.1 question answering Test F1
to 93.2 and SQuAD v2.0 Test F1 to 83.1.
""",
    },
    {
        "title": "rag_paper",
        "text": """
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Abstract:
Large pre-trained language models have been shown to store factual knowledge in their parameters,
and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their
ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-
intensive tasks, their performance lags behind task-specific architectures. Additionally, providing
provenance for their decisions and updating their world knowledge remain open research problems.

RAG Models:
We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models
which combine pre-trained parametric and non-parametric memory for language generation. Our RAG
models use a pre-trained seq2seq model as the parametric memory and a dense vector index of
Wikipedia as the non-parametric memory, accessed with a pre-trained neural retriever.

Retrieval Component:
The retrieval component is a bi-encoder architecture. The document encoder and query encoder
produce dense representations, and retrieval is performed using Maximum Inner Product Search (MIPS).
We use a pre-trained DPR model as the retriever.

Generation Component:
We feed the input sequence x and the retrieved document z into the generator, which produces the
final output y. The generator is a pre-trained BART-large model.

Results:
RAG models achieve state-of-the-art results on open-domain QA, surpassing parametric seq2seq
models and task-specific retrieve-and-extract architectures. RAG models also generate more
specific, diverse and factual language than a BART baseline.
""",
    },
]


def create_sample_data():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    for sample in SAMPLES:
        path = data_dir / f"{sample['title']}.txt"
        path.write_text(sample["text"].strip(), encoding="utf-8")
        print(f"✓ Created {path}")
    print(f"\nSample data ready in ./data/ ({len(SAMPLES)} files)")
    print("Now run: python ingest.py --data-dir data")


if __name__ == "__main__":
    create_sample_data()
