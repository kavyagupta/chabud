import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models.segmentation.deeplabv3 import IntermediateLayerGetter, DeepLabHead
from torchvision.models import resnet50, resnet101, mobilenet_v3_large



class DeepLabChangeDetectionModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        input_shape = x1.shape[-2:]
        # contract: features is a dict of tensors
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)
        
        x1 = features1["out"]
        x2 = features2["out"]
        out = torch.concat([x1, x2], dim=1)
        out = self.classifier(out)
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        
        return out
    
def bidate_deeplab_resnet50(n_channels=12, n_classes=2):
    backbone = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
    return_layers = {"layer4": "out"}

    conv1 = backbone.conv1
    backbone.conv1 = nn.Conv2d(in_channels=n_channels, 
                            out_channels=conv1.out_channels, 
                            kernel_size=conv1.kernel_size, stride=conv1.stride, 
                            padding=conv1.padding, bias=conv1.bias)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = DeepLabHead(2048 * 2, n_classes)
    model = DeepLabChangeDetectionModel(backbone, classifier)

    return model 

def bidate_deeplab_resnet101(n_channels=12, n_classes=2):
    backbone = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
    return_layers = {"layer4": "out"}

    conv1 = backbone.conv1
    backbone.conv1 = nn.Conv2d(in_channels=n_channels, 
                            out_channels=conv1.out_channels, 
                            kernel_size=conv1.kernel_size, stride=conv1.stride, 
                            padding=conv1.padding, bias=conv1.bias)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = DeepLabHead(2048 * 2, n_classes)
    model = DeepLabChangeDetectionModel(backbone, classifier)

    return model 

def bidate_deeplab_mobilenet_v3_large(n_channels=12, n_classes=2):
    backbone = mobilenet_v3_large(weights=None, dilated=True)
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    return_layers = {str(out_pos): "out"}

    conv1 = backbone[0][0]
    backbone[0][0] = nn.Conv2d(in_channels=n_channels, 
                            out_channels=conv1.out_channels, 
                            kernel_size=conv1.kernel_size, stride=conv1.stride, 
                            padding=conv1.padding, bias=conv1.bias)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = DeepLabHead(out_inplanes * 2, n_classes)
    model = DeepLabChangeDetectionModel(backbone, classifier)

    return model