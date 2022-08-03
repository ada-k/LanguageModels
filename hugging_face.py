from transformers import pipeline
import pandas as pd


text = """Hey Jack, I just wanted to flag something with you. Last week when you 
said that you didn't want to watch the move Twilight with me, even in jest, it kind 
of got under my skin. I mainly feel like it's hypocritical when you make me watch basketball
games with you and our main activity together is watching sports on TV. I just wanted to get 
it off my chest. From Sophie"""


"""sentiment classification"""

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
outputs = classifier(text)
print(pd.DataFrame(outputs))

"""named entity recognition"""

ner_tagger = pipeline("ner", aggregation_strategy="simple", 
                      model="dbmdz/bert-large-cased-finetuned-conll03-english")
outputs = ner_tagger(text)
print(pd.DataFrame(outputs))

"""question answering"""

reader = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What movie did Jack not watch?"
outputs = reader(question=question, context=text)
print(pd.DataFrame([outputs]))

"""summarisation"""
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

"""text generation"""
generator = pipeline("text-generation")
response = "Dear Sophie, I am sorry I didn't watch Twilight with you."
prompt = text + "\n\nJack apology to Sophie:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])


"""masked language model"""
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Artificial Intelligence [MASK] take over the world.")