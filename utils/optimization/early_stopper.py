import copy


class EarlyStopper:
    def __init__(self, patience, min_delta=0, mode="minimize"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.best_model_params = None
        self.counter = 0

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_params = copy.deepcopy(model.state_dict())
        elif (((self.mode == "minimize") and (score < self.best_score - self.min_delta)) or
              ((self.mode == "maximize") and (score > self.best_score + self.min_delta))):
            self.best_score = score
            self.best_model_params = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        else:
            return False

    def get_best_score(self):
        return self.best_score

    def get_best_model_params(self):
        return self.best_model_params

