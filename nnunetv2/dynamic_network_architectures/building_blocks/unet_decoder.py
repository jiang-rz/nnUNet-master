import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

class CsubjectFormer(nn.Module):
    def __init__(self,n_stages, features_per_stage,strides):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        self.n_stages=n_stages
        self.features_per_stage=features_per_stage
        #print(self.features_per_stage)#brats[32, 64, 128, 256, 320, 320] 128,64,32,16,8,4

        self.strides=strides
        self.num_heads=[features_per_stage[i]//16 for i in range(n_stages)]
        self.blocks = nn.ModuleList([
            basicformer_csubject(
                dim=features_per_stage[i],
                num_heads=self.num_heads[i],)
            for i in range(n_stages)])
    def forward(self,x,feature):
        if feature==None:
            return x
        res=[]
        weight_list=[]
        for i in range(self.n_stages-1):
            #print(feature[i].shape)
            if i>=0 and i<4:
                resi=self.blocks[i](feature[i],x[i])
                #weight_list.append(weight)
                #print('average weight',torch.mean(weight ))
                #with open('/data7/runzejiang/nnUNet_raw/Dataset047_Atlas/inferTs1/atlas_weight.text','a') as f:
                #   f.write(str(torch.mean(weight ))+"\n")
            else:
                resi=x[i]
            res.append(resi)
        res.append(x[-1])
        return res
from nnunetv2.dynamic_network_architectures.building_blocks.tokenlearner import TokenLearner,TokenAttention
class basicformer_csubject(nn.Module):
    def __init__(self,dim,num_heads):
        super().__init__()
        self.dim=dim
        self.num_heads = num_heads
        self.token_learner=TokenLearner(num_feature=dim,num_tokens=10)
        self.token_attention=Attention_csubject(dim=self.dim,num_heads=self.num_heads)
        self.cross_attention=Attention_csubject(dim=self.dim,num_heads=self.num_heads)
        #self.cross_attention2Ntoken=Attention_2Ntoken(dim=dim, num_heads=num_heads)
        #self.self_attention=Attention(dim=self.dim,num_heads=self.num_heads)
        self.norm1=nn.LayerNorm(self.dim)
        self.norm2=nn.LayerNorm(self.dim)
        self.norm3= nn.LayerNorm(self.dim)
        self.norm4 = nn.LayerNorm(self.dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim, act_layer=nn.GELU)
        self.mlp2 = Mlp(in_features=dim, hidden_features=dim, act_layer=nn.GELU)
        self.relu=nn.ReLU()
        #self.instancenorm=torch.nn.InstanceNorm3d(self.dim)
    def forward(self,feature_token,x,usefeature):
        b, c, h, w, z = x.shape

        x = x.view(b, h * w * z, c)
        x_input = self.norm1(x)

        #feature=self.norm2(self.self_attention(feature)+feature)
        #feature=self.norm3(self.mlp1(feature)+feature)
        x_token = self.token_learner(x_input)
        if usefeature==False:
            #x = self.norm2(x)
            x = x.view(b, c, h, w, z)
            return x,None,x_token
        if feature_token!= None:
            feature_token = feature_token.unsqueeze(0).repeat(b, 1, 1)
        #if feature_token == None and usefeature==True:
        #    return x.view(b,c,h,w,z),None,x_token
            ref_token=x_token
            #ref_token=torch.cat([x_token,feature_token],dim=1)
            feature,_=self.token_attention(ref_token,x_token)
            #weights = torch.cosine_similarity(x_token, feature, dim=2, eps=1e-08)
            #weights = self.relu(weights)
            #ref = (x_token + feature * weights.unsqueeze(2))
            ref=x_token+feature
            ref=self.mlp1(self.norm2(ref))+ref
            ref=self.norm3(ref)
        else:
            ref=x_token
        #print(feature.shape)


        x_csubject,_=self.cross_attention(ref,x_input)
        #x=0.5*x+0.5*self.cross_attention2Ntoken(x,x_csubject)
        #weights = torch.cosine_similarity(x, x_csubject, dim=2, eps=1e-08)
        #weights = self.relu(weights)
        #with open('/data7/runzejiang/nnUNet_raw/Dataset047_Atlas/inferTs1/atlas_weight.text', 'a') as f:
        #    f.write(str(torch.mean(weights ))+"\n")
        #print(torch.max(weights))
        # print(torch.max(weights),torch.min(weights))
        #x = (x + x_csubject * weights.unsqueeze(2))
        x=x+x_csubject
        x=self.mlp2(self.norm4(x))+x
        #x=x_csubject+shortcut
        # attn_windows = self.norm3(attn_windows_csubject )
        #x = self.norm2(x)
        #weights=weights.view(b,1,h,w,z)
        x=x.view(b,c,h,w,z)
        #x=self.instancenorm(x)
        return x,None,x_token


class GRU(nn.Module):
    def __init__(self,dim):
        self.dim=dim
        self.linear1=nn.Linear(dim*2,dim)
        self.linear2=nn.Linear(dim*2,dim)
        self.linear3=nn.Linear(dim*2,dim)
        self.sigmoid=torch.sigmoid()
        self.mlp=Mlp(in_features=dim*2,hidden_features=dim,out_features=dim)
    def forward(self,x,x_csubject):
        x_concat=torch.concat([x,x_csubject],-1)
        r=self.sigmoid(self.linear1(x_concat))
        z=self.sigmoid(self.linear2(x_concat))
        x_csubject_r=r*x_csubject
        x_concat_r=torch.concat([x,x_csubject_r],-1)
        x_return=(1-z)*x_csubject+z*self.mlp(x_concat_r)
        return x_return



class Attention_2Ntoken(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x, reference):
        b,n,c=x.shape
        if reference==None:
            return x
        #else:
        #    alltoken=torch.cat([x,reference],dim=1)
        #b,n,c=alltoken.shape
        kv_x = self.kv(x)
        kv_r=self.kv(reference)
        q = self.q(x)
        kv_x = kv_x.reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        kv_r=kv_r.reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k_x, v_x = kv_x[0], kv_x[1]#b,h,n,d
        k_r, v_r = kv_r[0], kv_r[1]
        q = q * self.scale
        attn_xx=torch.sum(q*k_x,dim=-1,keepdim=True)#b,h,n,1
        attn_xv=torch.sum(q*k_r,dim=-1,keepdim=True)
        attn=torch.cat([attn_xx,attn_xv],dim=-1)#b,h,n,2
        v=torch.cat([v_x.unsqueeze(3),v_r.unsqueeze(3)],dim=3)#b,h,n,2,d
        #attn = (q @ k.transpose(-2, -1).contiguous())
        attn = self.softmax(attn)
        #plot_heatmap(attn)
        attn = self.attn_drop(attn)
        #print(attn.shape,v.shape)
        x = torch.sum(attn.unsqueeze(4)*v,dim=3).reshape(b, n, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class Attention_csubject(nn.Module):
    def __init__(self, dim,  num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.norm1=nn.LayerNorm(dim)
    def forward(self, feature, x, plotflag=False,  csubject_mask=None):
        B_, N, C = feature.shape
        b_, n, c=x.shape
        kv = self.kv(feature)
        q = x

        kv = kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        l_att=0
        if csubject_mask is not None:
            # nW = mask.shape[0]
            attn0=self.softmax(attn)
            attn = attn.view(B_, self.num_heads, n, N) + csubject_mask.unsqueeze(1)
            attn = self.softmax(attn)
            #if c==768 and N>50 and plotflag==True:
            #    plot_heatmap(attn0,attn)
            l_att=torch.sum(torch.abs(attn0-attn))
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, n, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x,l_att
class Attention(nn.Module):
    def __init__(self, dim,  num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        self.norm1=nn.LayerNorm(dim)
    def forward(self, feature):
        N, C = feature.shape
        kv = self.kv(feature)
        q = self.q(feature)

        kv = kv.reshape( N, 2, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3).contiguous()
        q = q.reshape(N, self.num_heads, C // self.num_heads).permute(1, 0, 2).contiguous()
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 1).reshape(N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
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
class conv1(nn.Module):
    def __init__(self, in_channels, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, (1, 1, 1), stride=1, padding=0)
        self.act = act_layer()
        self.norm=nn.LayerNorm(in_channels)
    def forward(self, x):
        x=self.conv(x)
        x=self.act(x)+x
        b, c, h, w, z = x.shape
        x = x.view(b, h * w * z, c)
        x = self.norm(x)
        x = x.view(b, c, h, w, z)
        return x


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        #convlist=[]
        stages = []
        stages_2=[]
        transpconvs = []
        csubject_former=[]
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))
            #convlist.append(conv1(encoder.output_channels[-(s + 1)]))
            stages_2.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, 2*input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))


            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            csubject_former.append(basicformer_csubject(dim=input_features_skip,num_heads=input_features_skip//16))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        #self.convlist=nn.ModuleList(convlist)
        self.stages_2 = nn.ModuleList(stages_2)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.csubject_former = nn.ModuleList(csubject_former)
        #self.csubject_former = CsubjectFormer(n_stages, features_per_stage, strides)

    def forward(self, skips,feature=None,usefeature=False):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        feature_outputs=[]
        seg_outputs = []
        weights_list=[]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip=skips[-(s+2)]
            #feature_outputs.append(skip)
            #skip=self.convlist[s](skip)
            if feature != None:
                feature_s = feature[-1 - s]
            else:
                feature_s = None

            """skip1, weightss,feature_token = self.csubject_former[s](feature_s, skip,usefeature)
            feature_outputs.append(feature_token)
            weights_list.append(weightss)"""

            x = torch.cat((x, skip), 1)
            x = self.stages_2[s](x)
            x, weightss, feature_token = self.csubject_former[s](feature_s, x, usefeature)
            feature_outputs.append(feature_token)
            weights_list.append(weightss)

            x = self.stages[s](x)
            """x2, weightss, feature_token = self.csubject_former[s](feature_s, x, usefeature)
            feature_outputs.append(feature_token)
            weights_list.append(weightss)"""
            if self.deep_supervision:
                #fs = self.stages_2[s](fs)
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                #fs = self.stages_2[s](fs)
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        weights_list=weights_list[::-1]
        seg_outputs = seg_outputs[::-1]
        feature_outputs=feature_outputs[::-1]
        if not self.deep_supervision:
            r = weights_list[0],feature_outputs,seg_outputs[0]
        else:
            r = weights_list,feature_outputs,seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output