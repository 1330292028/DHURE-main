import torch
import torch.nn as nn
import numpy as np

class A_net(nn.Module):
    def __init__(self, num=64):
        super(A_net, self).__init__()
        self.A_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
            #nn.Tanh(),
        )

    def forward(self, input):
        #out = self.A_net(input)
        return torch.sigmoid(self.A_net(input)) 

class a_net(nn.Module):
    def __init__(self):
        super(a_net, self).__init__()
        self.A_net = A_net(num=64)
        
    def forward(self, input):
        #  junyun
        ones = torch.ones_like(input).cuda()
        b,c,h,w = input.shape
        pool = nn.MaxPool2d((h,w))
        #input_a = ones * (-pool(-input))
        input_a = ones * (pool(input))
#
        a = self.A_net(input_a)
        EL = torch.pow(input, a)
        # no junyun
#        ones = torch.ones_like(input).cuda()
#        b,c,h,w = input.shape
#        pool = nn.MaxPool2d((h,w))
#        max_l = pool(input)
#        min_l = (-pool(-input))
#        tv_l = max_l - min_l
#        x = ones * (min_l + 3.0/5.0*tv_l)
##        t1 = max_l
##        t2 = min_l + 9.0/10.0*tv_l
##        t3 = min_l + 8.0/10.0*tv_l
##        d1 = input > t2
##        d2 = (input<=t2) & (input>t3)
##        d3 = input <= t3
##        x1 = t1 * d1
##        x2 = t2 * d2
##        x3 = t3 * d3
##        x = x1+x2+x3
#
#        a = self.A_net(x)
#        EL = torch.pow(input, a)

        return EL, a

class aa_net(nn.Module):
    def __init__(self):
        super(aa_net, self).__init__()
        self.A_net = A_net(num=64)
        
    def forward(self, input):
        #  junyun
        ones = torch.ones_like(input).cuda()
        b,c,h,w = input.shape
        pool = nn.MaxPool2d((h,w))
        input_a = ones * (pool(input))
        a = self.A_net(input_a)
        EL = torch.pow(input, a)
        # no junyun
#        ones = torch.ones_like(input).cuda()
#        b,c,h,w = input.shape
#        pool = nn.MaxPool2d((h,w))
#        max_l = pool(input)
#        min_l = (-pool(-input))
#        tv_l = max_l - min_l
#        x = ones * (min_l + 3.0/5.0*tv_l)
##        t1 = max_l
##        t2 = min_l + 9.0/10.0*tv_l
##        t3 = min_l + 8.0/10.0*tv_l
##        d1 = input > t2
##        d2 = (input<=t2) & (input>t3)
##        d3 = input <= t3
##        x1 = t1 * d1
##        x2 = t2 * d2
##        x3 = t3 * d3
##        x = x1+x2+x3
#
#        a = self.A_net(x)
#        EL = torch.pow(input, a)

        return EL, a
