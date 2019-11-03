import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def CUDA(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    

class ConvBlock_mix(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock_mix, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def show(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = torch.mean(x, dim=1)
        return x
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

    
class Cnns(nn.Module):
    
    def __init__(self, classes_num=50, activation='logsoftmax'):
        super(Cnns, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock_mix(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock_mix(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock_mix(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock_mix(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        
    def show(self, input):
       
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        b ,c, d, e, f = self.conv_block1.show(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
        x1 = torch.mean(x, dim=1)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
        x2 = torch.mean(x, dim=1)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
        x3 = torch.mean(x, dim=1)
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')#（64，961，53，5）
        x4 = torch.mean(x, dim=1)
        return x1, x2, x3, x4, b, c, d, e, f
    
    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')#（64，64，215，20）
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')#（64，193，107，10）
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')#（64，449，53，5）
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')#（64，961，53，5）
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output  
class Crnn(nn.Module):
    def __init__(self, classes_num=2, activation='logsoftmax'):
        super(Crnn, self).__init__()
        self.channels = 1
        self.hidden_size = 64
        self.rnn_input_size = 128
        self.num_layers = 2
        self.conv_kernel_size = 32
        self.p = 0.3
        self.activation = activation
        
        self.branch1x1_1 = BasicConv2d(1, 32, kernel_size=1) 
        
        self.branch3x3_1 = BasicConv2d(1, 32, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(32,64, kernel_size=(1, 3), padding=(0, 1)) 
        self.branch3x3_3 = BasicConv2d(64,32, kernel_size=(3, 1), padding=(1, 0)) 
        
        self.branch5x5_1 = BasicConv2d(1, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32,64, kernel_size=(1, 5), padding=(0, 2))      
        self.branch5x5_3 = BasicConv2d(64,32, kernel_size=(5, 1), padding=(2, 0)) 
              
        self.branch7x7_1 = BasicConv2d(1, 32, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(32,64, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(64,32, kernel_size=(7, 1), padding=(3, 0))       


        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 64), stride=1)#97

        # rnn-gru layer
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=self.p
        )

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.p),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, 2),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size, hidden_size):
        return CUDA(Variable(torch.zeros(self.num_layers*2, batch_size, hidden_size)))

    def forward(self, x):
        x = x.unsqueeze(1)
        batch_size = x.size(0)
       
        branch1x1 = self.branch1x1_1(x)       
        

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)
               
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        outputs = (branch1x1,branch3x3,branch5x5,branch7x7)
        feature = torch.cat(outputs, 1)
        output = self.max_pool(feature).squeeze(-1).permute(0,2,1)
        h_state = self.init_hidden(batch_size, self.hidden_size)
        self.rnn.flatten_parameters()
        output, h_state = self.rnn(output, h_state)
        output = self.fc(output[:, :, self.hidden_size:]).squeeze(0)
        return output