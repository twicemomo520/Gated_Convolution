import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import yaml 
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from loss import perceptual_loss
from network import generator, discriminator



class BaseModel(nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()
        self.name = name
        
    def save(self, epoch, train_loss, val_loss, trainloss_array, valloss_array):
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.trainloss_array = trainloss_array
        self.valloss_array = valloss_array
        
        self.checkpoint_name = 'model_epoch{}_traloss{}_valloss{}.pth'.format(self.epoch, self.train_loss, self.val_loss)
        self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.checkpoint_name)
        print('now save checkpoint to {} ....'.format(self.checkpoint_path))
        
        if not os.path.exists(self.config.checkpoint_path):
            os.mkdir(self.config.checkpoint_path)
        checkpoinnt = {'epoch': self.epoch,
                       'loss': self.train_loss,
                       'trainloss_array': self.trainloss_array,
                       'valloss_array': self.valloss_array,
                       'generator': self.generator.state_dict(),
                       'discriminator': self.discriminator.state_dict()} 
        torch.save(checkpoinnt, self.checkpoint_path)
        print('save checkpoint complete')    
        
    def load(self, choose_checkpoint):   
        self.checkpoint_name = choose_checkpoint        
        if os.path.exists(self.config.checkpoint_path):
            self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.checkpoint_name)
            print('load checkpoint of {} ....'.format(self.checkpoint_path))
            data = torch.load(self.checkpoint_path)
            self.epoch = data['epoch']
            self.train_loss = data['loss']
            self.trainloss_array = data['trainloss_array']
            self.valloss_array = data['valloss_array']
            self.generator.load_state_dict(data['generator'])
            self.discriminator.load_state_dict(data['discriminator'])
        print('load checkpoint complete')    
        
        return self.epoch, self.train_loss, self.trainloss_array, self.valloss_array
        

class inpaintingModel(BaseModel):
    def __init__(self, config_path):        
        super(inpaintingModel, self).__init__('inpaintingModel')
        
        self.config = load_config(config_path)
        self.generator = generator(in_channel = 4, init_weights=True)  
        self.discriminator = discriminator(use_spectral_norm = True, init_weights=True)                      
        
        
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        percept_loss = perceptual_loss()
        
        self.add_module('l1_loss', l1_loss)
        self.add_module('mse_loss', mse_loss)
        self.add_module('percept_loss', percept_loss)
    
        #self.gen_optimizer = optim.SGD(params = self.generator.parameters(), lr= self.config.LR, momentum=0.9)
        #self.dis_optimizer = optim.SGD(params = self.discriminator.parameters(), lr= self.config.LR, momentum=0.9)

        self.gen_optimizer = optim.RMSprop(params = self.generator.parameters(), lr= self.config.LR, momentum=0.9, alpha = 0.9, eps = 0.001)
        self.dis_optimizer = optim.RMSprop(params = self.discriminator.parameters(), lr= self.config.LR, momentum=0.9, alpha = 0.9, eps = 0.001)
        
    def process(self, images, masks, is_backward):
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        
        first_output, outputs = self(images, masks)
        
        gen_loss = 0
        dis_loss = 0
        
        composed_images = outputs * masks + images * (1-masks)
        
        dis_real_input = torch.cat((images, masks), dim = 1)
        dis_fake_input = torch.cat((composed_images.detach(), masks), dim = 1)
        real_labels = self.discriminator(dis_real_input)
        fake_labels = self.discriminator(dis_fake_input)
        
        valid = torch.Tensor(np.ones(real_labels.shape)).to(torch.device('cuda:0'), torch.float)
        fake = torch.Tensor(np.zeros(real_labels.shape)).to(torch.device('cuda:0'), torch.float)
        
        real_loss = self.mse_loss(real_labels, valid)
        fake_loss = self.mse_loss(fake_labels, fake)
        dis_loss = dis_loss + (real_loss + fake_loss) / 2
        
        if is_backward:
             dis_loss.backward()
             self.dis_optimizer.step()            
        #####################################################
        
        
        composed_images = outputs * masks + images * (1-masks)
        
        gen_fake_input = torch.cat((composed_images, masks), dim = 1)
        gen_fake_labels = self.discriminator(gen_fake_input)             
        valid = torch.Tensor(np.ones(gen_fake_labels.shape)).to(torch.device('cuda:0'), torch.float)
        
        gen_gan_loss = self.mse_loss(gen_fake_labels, valid)
        gen_loss += 0.1*gen_gan_loss
        
        l1_loss = self.l1_loss(composed_images, images)
        gen_loss += 10*l1_loss
        
        percept_loss = self.percept_loss(images, outputs, masks)
        gen_loss += percept_loss
        
        
        
        if is_backward:
            gen_loss.backward()
            self.gen_optimizer.step()            
        
        ######################################################
        return outputs, dis_loss, gen_loss
    
    def forward(self, images, masks):
        image_masked = (images * (1-masks)) + masks
        first_output, second_output = self.generator(images, masks)
        return first_output, second_output
               
    
class load_config():
    def __init__(self, path):
        with open(path, 'r', encoding = 'utf-8') as stream:
            config = yaml.load(stream)
        self.config = config 

    def __getattr__(self, name):
        return self.config[name]










'''
from torch.utils.data import DataLoader, Dataset
from dataset import outdoordata 
train_data = outdoordata('./data/outdoor2_data', './data/mask')
train_loader = DataLoader(dataset = train_data,
                          batch_size = 2,
                          num_workers = 0,
                          drop_last = True,
                          shuffle = True)  

model = inpaintingModel('./config.yml').to(torch.device('cuda:0'), dtype = torch.float)

'''

'''

for i, items in enumerate(train_loader):
    images = items[0].to(torch.device('cuda:0'), torch.float)
    masks = items[1].to(torch.device('cuda:0'), torch.float)
    outputs, dis_loss, gen_loss = model.process(images, masks, False)
    loss = (dis_loss + gen_loss) / 2
    break

'''

