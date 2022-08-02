# LanguageModels

Use cases:
1. Text generation.
2. Text classification/Sentiment Analysis.
3. Text Summarisation.
4. Text rewriting/Paraphrasing.
5. Text clustering.
6. Embeddings generation.
7. 

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

[ULMFiT](ulmfit.py)


[Transformer](transformer.py)


[Google’s BERT](bert.py)






## Word Embeddings
ELMo
Flair

## Other Pretrained Models
StanfordNLP
 
