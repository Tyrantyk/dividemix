

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            # if param.requires_grad:
            self.shadow[name] = param.data.detach().clone()

    def update(self, model):
        for name, param in model.named_parameters():
            # if param.requires_grad:
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.detach().clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            # if param.requires_grad:
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            # if param.requires_grad:
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}
