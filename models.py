from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class Roberta:
    @staticmethod
    def load_models(tasks):
        """Load all models"""
        MODELS = {}
        for task in tasks:
            model_path = f"cardiffnlp/twitter-roberta-base-{task}"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            MODELS[task] = [tokenizer, model]
        return MODELS

    TASKS = ["emotion", "hate", "irony", "offensive", "sentiment"]
    MODELS = load_models(TASKS)

    @staticmethod
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    @staticmethod
    def prediction(text: str, task: str):
        tokenizer, model = Roberta.MODELS[task]

        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        label = output[0][0].detach().numpy().argmax(axis=0)
        return label

    @staticmethod
    def predictions(text: str, task: str | None = None):
        """
        if task n'est pas déf, on retourne les prédictions de tous les modèles
        """
        preprocessedText = Roberta.preprocess(text)
        if task is None:
            tasks = ["emotion", "hate", "irony", "offensive", "sentiment"]
            labelsDicts = {}
            for task in tasks:
                label = Roberta.prediction(preprocessedText, task)
                labelsDicts[task] = int(label)
            return labelsDicts
        labelDict = {}
        label = Roberta.prediction(preprocessedText, task)
        labelDict[task] = int(label)
        return labelDict


if __name__ == "__main__":

    print(Roberta.predictions("lol"))
