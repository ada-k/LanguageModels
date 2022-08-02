# LanguageModels

Use cases:
1. Text generation.
2. Text classification/Sentiment Analysis.
3. Text Summarisation.
4. Text rewriting/Paraphrasing.
5. Text clustering.
6. Embeddings generation.
7. Translation.
8. 

## Multi-Purpose NLP Models

[Transformer-XL](transformer_xl.py)
- Text generation steps:
```
python pytorch-transformers/examples/run_generation.py 
    --model_type=transfo-xl 
    --length=100 
    --model_name_or_path=transfo-xl-wt103
```

[XLNet](xlnet.py)
- Text generation steps:
```
python pytorch-transformers/examples/run_generation.py
    --model_type=xlnet
    --length=50
    --model_name_or_path=xlnet-base-cased
```


[OpenAI’s GPT-2](gpt2.py)
- Text completion steps:
1. Tokenize and index the text as a sequence of numbers
2. Pass it to the gp2 pretrained model e.g Pytorch's `GPT2LMHeadModel`. 
3. Get predictions.

- Text generation steps:
```
python pytorch-transformers/examples/run_generation.py
    --model_type=gpt2
    --length=100
    --model_name_or_path=gpt2
```

[Universal Language Model Fine Tuning - ULMFiT](ulmfit.py)\
Steps:
1. Data prep.
2. Creating LM Model & fine-tuning it with the pre-trained model.
3. Get predictions with the fine tuned model.

Implementations in Spacy and Fastai.

[Transformer](transformer.py)


[Google’s BERT](bert.py)
- Masked language modeling steps:
1. Text tokenisation.
2. Convert tokesn into a sequence of integers.
3. Use bert's masked language model e.g Pytorch's `BertForMaskedLM`.
4. Get predictions.



## Word Embeddings
[Embeddings from Language Model - ELMo](elmo.py)
- NLP framework by AllenNLP. Word vectors are calculated using a 2-layer bidirectional language model (biLM). Each layer comprises back &forward pass.
- Represents word embeddings using complete sentence, thus, capture the context of the word used in the sentence unlike Glove and Word2Vec.


[Flair](flair.py)
-  Captures latent syntattic-semantic info from text.
-  Gives word embeddings based on its sorrounding text.

