import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
class TokenLearner(nn.Module):
    def __init__(self,num_feature,num_tokens,):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_feature=num_feature
        self.mlp=Mlp(num_feature, num_feature, num_tokens)
        #self.norm=nn.LayerNorm(num_feature)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        #b,c,h,w,z=x.shape
        #x_reshape=x.view(b, h * w * z, c)
        #x_reshape=self.norm(x_reshape)
        x_mlp=self.mlp(x) #b , h *w*z ,num_token
        x_att=self.softmax(x_mlp.transpose(-2, -1).contiguous())#b,num_token,h*w*z
        x_input=x
        x_return =x_att@x_input#b,num_token,c
        return x_return

class TokenAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x, reference):
        _,n_x,_=x.shape
        if reference==None:
            alltoken=x
        else:
            alltoken=torch.cat([x,reference],dim=1)
        b,n,c=alltoken.shape
        kv = self.kv(alltoken)
        q = self.q(alltoken)
        kv = kv.reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        attn = self.softmax(attn)
        #plot_heatmap(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x[:,:n_x,:]

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
def generate_random_str(randomlength=16):
    """
    生成一个指定长度的随机字符串
      """
    random_str =''
    base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length =len(base_str) -1
    for i in range(randomlength):
        random_str +=base_str[random.randint(0, length)]
    return random_str

def plot_heatmap(attn):
    b,h,n,N=attn.shape
    print(torch.max(attn),torch.mean(attn))
    attn=(100*torch.sum(attn,dim=1,keepdim=False)).cpu().detach().int().numpy()
    print(attn.shape,np.mean(attn),np.max(attn))
    annot_kws = {"fontsize": 2}
    randomstr=generate_random_str(3)
    for i in range(b):

        ax1 = sns.heatmap(attn[i,:50,:50], annot=True,fmt="", annot_kws=annot_kws)
        ax1.set_title('attn ')
        ax1.set_xlabel('other subjects')
        ax1.set_ylabel('image patch')
        figure = ax1.get_figure()
        figure.savefig('/data7/runzejiang/nnUNet_results/Dataset049_Islesdwi/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/att_png/att'+randomstr+'.png', dpi=800,bbox_inches='tight')
        plt.clf()