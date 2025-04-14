import torch.nn as nn

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def forward_train(self, image, target, **kwargs):
        """Forward computation during training."""
        raise NotImplementedError

    def forward_test(self, image, **kwargs):
        """Forward computation during testing."""
        return self.student(image)

    def forward(self, image, target=None, **kwargs):
        if target is None:
            return self.forward_test(image, **kwargs)
        return self.forward_train(image, target, **kwargs)

    def get_learnable_parameters(self):
        """Returns all parameters that require gradient updates."""
        return [v for v in self.student.parameters() if v.requires_grad]

    def get_extra_parameters(self):
        """Returns parameters introduced by distillation (if any)."""
        return [v for v in self.parameters() if v.requires_grad and v not in self.student.parameters()]

    def train_step(self, image, target, **kwargs):
        """Performs a single training step."""
        output, losses_dict = self.forward_train(image, target, **kwargs)
        return output, losses_dict 