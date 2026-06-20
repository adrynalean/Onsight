import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os 
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def __init__(self):
        # Lazy: only load the (GPU-heavy) spaCy transformer model when we
        # actually run NER. Reading a precomputed stub needs no model.
        self.nlp_model = None

    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp

    def _model(self):
        if self.nlp_model is None:
            self.nlp_model = self.load_model()
        return self.nlp_model

    def get_ners_inference(self,script):
        nlp = self._model()
        script_sentences = sent_tokenize(script)

        ner_output = []

        for sentence in script_sentences:
            doc = nlp(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ =="PERSON":
                    # Use the LAST token, not the first: One Piece names like
                    # "Monkey D. Luffy", "Monkey D. Garp", "Monkey D. Dragon"
                    # all share the first token, but their last token (the given
                    # name) is the distinct, commonly-used handle. Same for
                    # "Roronoa Zoro" -> "Zoro", "Tony Tony Chopper" -> "Chopper".
                    tokens = entity.text.split()
                    if not tokens:
                        continue
                    name = tokens[-1].strip(".,!?;:'\"")
                    if name:
                        ners.add(name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self,dataset_path,save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
            return df

        # load dataset 
        df = load_subtitles_dataset(dataset_path)

        # Run Inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path,index=False)
        
        return df