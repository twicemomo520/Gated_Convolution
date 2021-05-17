import torch 
import torch.nn as nn
import cv2

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        
        

class gated_layer(BaseNetwork):
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation = 1, padding_mode = 'replicate', init_weights=True, use_spectral_norm=True, activation = 'lrelu', norm = 'in'):
        super(gated_layer, self).__init__()
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)         
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None 

            
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_channel)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_channel)            
        if norm == 'ln':
            self.norm = nn.LayerNorm(output_channel)                   
        if norm == 'none':
            self.norm = None    
            
        self.conv2d = spectral_norm(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, dilation, bias = not use_spectral_norm), use_spectral_norm)
        self.mask_conv2d = spectral_norm(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, dilation, bias = not use_spectral_norm), use_spectral_norm)
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        
        return self.sigmoid(mask)
        
    def forward(self, x):
        image_output = self.conv2d(x)
        mask_output = self.mask_conv2d(x)
        
        x = image_output * self.gated(mask_output)        
        
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
            
        return x    




class generator(BaseNetwork):
    def __init__(self, in_channel = 4, init_weights = True):
        super(generator, self).__init__()

        p1 = ((64 - 1)*1 - 64 + 2*(3 - 1) + 1) // 2
        p2 = ((64 - 1)*1 - 64 + 4*(3 - 1) + 1) // 2
        p3 = ((64 - 1)*1 - 64 + 8*(3 - 1) + 1) // 2
        p4 = ((64 - 1)*1 - 64 + 16*(3 - 1) + 1) // 2
        
        self.corse_net = nn.Sequential(  
            gated_layer(in_channel, 32, 5, stride = 1, padding = 2),#outsize = 256
        
            gated_layer(32, 64, 4, stride = 2, padding = 1),#outsize = 128
            gated_layer(64, 64, 3, stride = 1, padding = 1),#outsize = 128
        
            gated_layer(64, 128, 4, stride = 2, padding = 1),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64
        
            # p = (outsize - 1) * S - inputsize + dilation * (k - 1) +1
            gated_layer(128, 128, 3, stride = 1, padding = p1, dilation = 2),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = p2, dilation = 4),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = p3, dilation = 8),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = p4, dilation = 16),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64
        
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode = 'bilinear'),#outsize = 128
                gated_layer(128, 64, 3, stride = 1, padding = 1)), #outsize = 128
            gated_layer(64, 64, 3, stride = 1, padding = 1),#outsize = 128
        
        
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode = 'bilinear'),#outsize = 256
                gated_layer(64, 32, 3, stride = 1, padding = 1)), #outsize = 256
            gated_layer(32, 16, 3, stride = 1, padding = 1),#outsize = 256
         
            gated_layer(16, 3, 3, stride = 1, padding = 1, activation = 'sigmoid', norm = 'none')#outsize = 256
            )
        
        
        self.refine_net = nn.Sequential(
            gated_layer(in_channel, 32, 5, stride = 1, padding = 2),#outsize = 256
            
            gated_layer(32, 32, 4, stride = 2, padding = 1),#outsize = 128
            gated_layer(32, 64, 3, stride = 1, padding = 1),#outsize = 128
    
            gated_layer(64, 64, 4, stride = 2, padding = 1),#outsize = 64 
            gated_layer(64, 128, 3, stride = 1, padding = 1),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64                
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64 
            gated_layer(128, 128, 3, stride = 1, padding = p1, dilation = 2),#outsize = 64 
            gated_layer(128, 128, 3, stride = 1, padding = p2, dilation = 4),#outsize = 64 
            gated_layer(128, 128, 3, stride = 1, padding = p3, dilation = 8),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = p4, dilation = 16),#outsize = 64 
            
            )
        self.refine_attn = Self_Attn(128, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(        
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64
            gated_layer(128, 128, 3, stride = 1, padding = 1),#outsize = 64
    
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode = 'bilinear'),#outsize = 128
                gated_layer(128, 64, 3, stride = 1, padding = 1)), #outsize = 128
            gated_layer(64, 64, 3, stride = 1, padding = 1),#outsize = 128
            
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode = 'bilinear'),#outsize = 256
                gated_layer(64, 32, 3, stride = 1, padding = 1)), #outsize = 256
            gated_layer(32, 16, 3, stride = 1, padding = 1),#outsize = 256        
            
            gated_layer(16, 3, 3, stride = 1, padding = 1, activation = 'sigmoid', norm = 'none'),#outsize = 256 
    
            )    
        
        if init_weights:
            self.init_weights()
    
    def forward(self, image, mask):
        masked_iamge = image * (1-mask) + mask
        first_input = torch.cat([masked_iamge, mask], dim = 1)
        first_output = self.corse_net(first_input)
        
        second_masked_image = image * (1-mask) + first_output * mask
        second_input = torch.cat([second_masked_image, mask], dim = 1)  
        x = self.refine_net(second_input)
        x = self.refine_attn(x)
        second_output = self.refine_upsample_net(x)
        
        return first_output, second_output




class discriminator(BaseNetwork):
    def __init__(self, use_spectral_norm = True, init_weights=True):
        super(discriminator, self).__init__()
        
        self.block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, 7, 1, 3, padding_mode = 'reflect'), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace = True))
        
        self.block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, padding_mode = 'reflect'), use_spectral_norm),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True))

        self.block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, padding_mode = 'reflect'), use_spectral_norm),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True))


        self.block4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, 4, 2, 1, padding_mode = 'reflect'), use_spectral_norm),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True))


        self.block5 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, 4, 2, 1, padding_mode = 'reflect'), use_spectral_norm),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True))        

        self.block6 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 1, 4, 2, 1, padding_mode = 'reflect'), use_spectral_norm))
   
        
        if init_weights:
            self.init_weights()
            

    def forward(self, x):                  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        return x






def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module





class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out














