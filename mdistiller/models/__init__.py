from .cifar import cifar_model_dict, tiny_imagenet_model_dict
from .imagenet import imagenet_model_dict
from .resnet import resnet20, resnet110

model_dict = {
    "resnet20": resnet20,
    "resnet110": resnet110,
}
