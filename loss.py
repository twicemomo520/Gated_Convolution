import torchvision.models as models
import torch
import torch.nn as nn 

'''
model = models.vgg16(pretrained=True)
x = torch.randn((1,3,256,256))

model.features.parameters()

output = model.features(x)
for i in model.features[0].parameters():
    print(i)
    print(i.shape)
'''

class VGG16_extractor(nn.Module):
    def __init__(self):
        super(VGG16_extractor, self).__init__()
        
        model  = models.vgg16(pretrained = True)
        model.eval()
        self.extract_1 = model.features[:5]
        self.extract_2 = model.features[5:10]
        self.extract_3 = model.features[10:17]
    
        for i in range(3):
            for param in getattr(self, 'extract_{}'.format(i+1)).parameters():
                param.requires_grad = False
            
    def forward(self, image):    
        x1 = self.extract_1(image)
        x2 = self.extract_2(x1)
        x3 = self.extract_3(x2)
        
        return [x1, x2, x3]
        
class perceptual_loss(nn.Module):
    def __init__(self):
        super(perceptual_loss, self).__init__()
        
        self.vgg_extractor = VGG16_extractor()
        self.L1loss = nn.L1Loss()
        
    def forward(self, ori_image, output_image, mask):
        composed_image = output_image * mask + ori_image * (1-mask)
        
        composed_feature = self.vgg_extractor(composed_image)
        ori_feature = self.vgg_extractor(ori_image)
        output_feature = self.vgg_extractor(output_image)
        
        loss = 0
        
        for i in range(len(composed_feature)):
            loss += self.L1loss(composed_feature[i], ori_feature[i])
            loss += self.L1loss(output_feature[i], ori_feature[i])
        
        return loss    
        










import torch
import torch.nn as nn
import torchvision.models as models


class Adversarial_loss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(Adversarial_loss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss