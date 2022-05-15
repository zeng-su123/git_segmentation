from .custom_pspnet import *
from .pspnet import *
from .resnet import *
from .efficientnet import efficientnet_model_selector
from .small_segmentation_models import small_segmentation_model_selector


# 备选的网络有pspnet , resenet34 ,resnet18 , efficientnet

def model_selector(model_name, num_classes=4, in_channels=1):
    """

    :param model_name:
    :param num_classes:
    :param in_channels:
    :return:
    """
    classification = False
    if "classification" in model_name:  # classification 有什么用？
        classification = True

    # if "small_segmentation" in model_name:
    #     return small_segmentation_model_selector(model_name, num_classes)
    # if "custom_pspnet" in model_name:
    #     return custom_psp_model_selector(model_name, num_classes)

    if "pspnet" in model_name:  # pspnet 也是一种分割模型
        return psp_model_selector(model_name, num_classes, classification)

    if "resnet34" in model_name or "resnet18" in model_name:
        return resnet_model_selector(model_name, num_classes, classification, in_channels)

    if "efficientnet" in model_name:
        return efficientnet_model_selector(model_name, num_classes, classification, in_channels)

    assert False, "Unknown model selected: {}".format(model_name)
