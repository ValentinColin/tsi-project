from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class Roberta:
    def __init__(self):
        self.tasks = ["emotion", "hate", "irony", "offensive", "sentiment"]

    @staticmethod
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    @staticmethod
    def load_model(task: str):
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        return [tokenizer, model]

    @staticmethod
    def prediction(text: str, task: str):
        tokenizer, model = Roberta.load_model(task)

        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        label = output[0][0].detach().numpy().argmax(axis=0)
        return label

    def predictions(self, text: str, task: str | None = None):
        """
        if task n'est pas déf, on retourne les prédictions de tous les modèles
        """
        preprocessedText = self.preprocess(text)
        if task is None:
            labelsDicts = {}
            for task in self.tasks:
                label = self.prediction(preprocessedText, task)
                labelsDicts[task] = int(label)
            return labelsDicts
        labelDict = {}
        label = self.prediction(preprocessedText, task)
        labelDict[task] = int(label)
        return labelDict


if __name__ == "__main__":
    Roberta_models = Roberta()
    print(Roberta_models.prediction(text="lol", task="sentiment-latest"))
    print(Roberta_models.predictions("lol"))
