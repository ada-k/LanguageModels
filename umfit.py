## Fast AI

from fastai.text import *
from fastai import *


"""Language model"""
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)
# train the learner object with learning rate = 1e-2
learn.fit_one_cycle(1, 1e-2)
learn.save_encoder('ft_enc')

"""classifier model"""
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
learn = text_classifier_learner(data_clas, drop_mult=0.7)
learn.load_encoder('ft_enc')
learn.fit_one_cycle(1, 1e-2)



# get predictions
preds, targets = learn.get_preds()

predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)


# spacy
from spacy_transformers import Transformer, TransformerModel
from spacy_transformers.annotation_setters import null_annotation_setter
from spacy_transformers.span_getters import get_doc_spans

trf = Transformer(
    nlp.vocab,
    TransformerModel(
        "bert-base-cased",
        get_spans=get_doc_spans,
        tokenizer_config={"use_fast": True},
    ),
    set_extra_annotations=null_annotation_setter,
    max_batch_items=4096,
)