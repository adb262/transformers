import torch
from sklearn import metrics

class ModelEval:
    def __init__(self, model, dataloader):
        self.model = model
        self.model.eval()

        self.dataloader = dataloader
        self.labels = []
        self.predictions = []
    
    def getPredictions(self) -> None:
        with torch.no_grad():
            for idx, (labels, text) in enumerate(dataloader):
                self.labels.extend(labels)
                self.predictions.extend(model(text))

    def getConfusionMatrix(self) -> metrics.ConfusionMatrix:
        return metrics.confusion_matrix(self.labels, self.predictions)


