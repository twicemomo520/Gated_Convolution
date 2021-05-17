import os
import re
import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
import glob
import pathlib
from torch.utils.data import DataLoader, Dataset
from dataset import outdoordata 
from model import inpaintingModel
from PIL import Image, ImageDraw

class trainer():
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.device = torch.device('cuda:0')
        self.model = inpaintingModel(config_path).to(self.device, dtype = torch.float)
        self.train_dataset = outdoordata(self.config.traindata_path, self.config.mask_path)
        self.val_dataset = outdoordata(self.config.valdata_path, self.config.mask_path)
        self.test_dataset = outdoordata(self.config.testdata_path, self.config.mask_path)
        
        self.delete_checkpoint = delete_bad_checkpoint(self.config.checkpoint_path)
        
        self.epoch = 0
        self.train_loss = 0
        self.val_loss = 0
        self.trainloss_array = []
        self.valloss_array = []
        
    def load(self):
        items = self.model.load(self.config.choose_checkpoint)
        self.epoch = items[0]
        self.train_loss = items[1]
        self.trainloss_array = items[2]
        self.valloss_array = items[3]
        
    def save(self):
        self.model.save(self.epoch, self.train_loss, self.val_loss, self.trainloss_array, self.valloss_array)
    
    def train(self):
        train_loader = DataLoader(dataset = self.train_dataset,
                                  batch_size = self.config.batchsize,
                                  num_workers = 0,
                                  drop_last = True,
                                  shuffle = True)       
        while(True):
            self.model.train()
            self.epoch += 1 
            iteration = 0
            iteration_loss = []
            
            for i, items in enumerate(train_loader):
                iteration += 1
                images = items[0].to(self.device, torch.float)
                masks = items[1].to(self.device, torch.float)
                outputs, dis_loss, gen_loss = self.model.process(images, masks, True)
                loss = (dis_loss + gen_loss) / 2
                iteration_loss.append(loss.item())
                print('epoch: {}, iteration: {}, loss:{}'.format(self.epoch, iteration, loss))
                
                if iteration % 200 == 0:
                    self.test()
            
            loss_mean = np.array(iteration_loss).mean()
            self.train_loss = loss_mean.item()
            self.trainloss_array.append(loss_mean.item())
            print('complete NO.{} training'.format(self.epoch))  
    
            
            self.valloss_array, self.val_loss = self.val()
            print('complete val')        
            self.save()
            
            draw_loss(self.trainloss_array, self.valloss_array)
            print('complete draw training and validation loss')
            
            self.delete_checkpoint.remove()
            print('remove bad checkpoint complete')    
            
            
    
    def val(self):
        self.model.eval()
        with torch.no_grad():
            val_loader = DataLoader(dataset = self.val_dataset,
                                  batch_size = self.config.batchsize,
                                  num_workers = 0,
                                  drop_last = True,
                                  shuffle = True)        
        
            val_loss = 0
            iteration = 0
            iteration_loss = []
            
            for i, items in enumerate(val_loader):
                iteration += 1
                images = items[0].to(self.device, torch.float)
                masks = items[1].to(self.device, torch.float)
                
                outputs, dis_loss, gen_loss = self.model.process(images, masks, False)
                
                loss = (dis_loss + gen_loss) / 2
                
                iteration_loss.append(loss.item())
                print('iteration: {}'.format(iteration))
            
            loss_mean = np.array(iteration_loss).mean()
            val_loss = loss_mean.item()
            self.valloss_array.append(val_loss)
            
            return self.valloss_array, val_loss
        
    def test(self):     
        
        total_number = len(glob.glob(os.path.join(self.config.testoutput_path, '*')))
        
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(dataset = self.val_dataset,
                              batch_size = 12,
                              num_workers = 0,
                              drop_last = True,
                              shuffle = True) 
            
            for x, items in enumerate(test_loader):
                images = items[0].to(self.device, torch.float)
                masks = items[1].to(self.device, torch.float)
                
                outputs, dis_loss, gen_loss = self.model.process(images, masks, False)
                
                masked_images = images * (1 - masks) + masks
                images = self.stitch_images(
                        self.postprocess(images),
                        self.postprocess(masked_images),
                        self.postprocess(outputs)
                        )
                
                name = 'epoch{}_{}.png'.format(self.epoch, str(total_number + 1))
                path = os.path.join(self.config.testoutput_path, name)
                images.save(path)
                
                print('第{}張圖片已完成'.format(x))
                
                break
            
    def stitch_images(self, inputs, *outputs, img_per_row=2):
        gap = 5
        columns = len(outputs) + 1

        width, height = inputs[0][:, :, 0].shape
        img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
        images = [inputs, *outputs]

        for ix in range(len(inputs)):
            xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
            yoffset = int(ix / img_per_row) * height

            for cat in range(len(images)):
                im = np.array((images[cat][ix]).cpu().data.numpy()).astype(np.uint8)[:,:,::-1].squeeze()
                im = Image.fromarray(im)
                img.paste(im, (xoffset + cat * width, yoffset))

        return img       

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img              
            
        
class draw_loss():
    def __init__(self, trainloss_array, valloss_array):
        plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
        total_epoch = len(trainloss_array)
        
        epoch_array = []
        for i in range(total_epoch):
            epoch_array.append(i+1)
            
        plt.plot(epoch_array, trainloss_array, 's-', color = 'r', label="train", linewidth=5.0)
        plt.plot(epoch_array, valloss_array, 'o-', color = 'g', label="val", linewidth=5.0)
        plt.title("loss")
        plt.show()
        
        
        
class load_config():
    def __init__(self, path):
        with open(path, 'r', encoding = 'utf-8') as stream:
            config = yaml.load(stream)
        self.config = config 

    def __getattr__(self, name):
        return self.config[name]
    
    
class delete_bad_checkpoint():
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
    
    def remove(self):
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_path, '*.pth'))
        while len(checkpoint_files) > 5: 
            valloss_array = []
            for i, files in enumerate(checkpoint_files):
                valloss = float(checkpoint_files[i].split('\\')[1].split('_traloss')[1].split('_valloss')[1].split('.pth')[0])
                valloss_array.append(valloss)
            index = valloss_array.index(max(valloss_array))
            os.remove(checkpoint_files[index])
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_path, '*.pth'))
        
        


if __name__ == '__main__':  
    trainer = trainer('./config.yml')
    output = trainer.train()
    
















'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


def main():
    for i in range(500):
        number = len(glob.glob('C:/Users/123/Desktop/gated convolution/data/mask/*'))
        # 1.创建白色背景图片
        d = 256
        img = np.zeros((d, d, 3), np.uint8) * 255
    
        # 2.循环随机绘制实心圆
        for i in range(0, 10):
            # 随机中心点
            center_x = np.random.randint(0, high=d)
            center_y = np.random.randint(0, high=d)
    
            # 随机半径与颜色
            radius = np.random.randint(5, high=d/5)
    
            cv2.circle(img, (center_x, center_y), radius, [255,255,255], -1)
        # 3.显示结果
        plt.imshow(img, 'gray')
        cv2.imwrite('C:/Users/123/Desktop/gated convolution/data/mask/{}circle_mask.png'.format(number + 1), img)


if __name__ == '__main__':
    main()

'''