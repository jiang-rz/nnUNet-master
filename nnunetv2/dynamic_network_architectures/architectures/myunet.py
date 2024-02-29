from typing import Union, Type, List, Tuple

import torch
from nnunetv2.dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from nnunetv2.dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

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

class basicformer_csubject(nn.Module):
    def __init__(self,dim,num_heads):
        super().__init__()
        self.dim=dim
        self.num_heads = num_heads
        self.attention=Attention_csubject(dim=self.dim,num_heads=self.num_heads)
        self.norm2=nn.LayerNorm(self.dim)
        #self.mlp = Mlp(in_features=dim, hidden_features=4*dim, act_layer=nn.GELU)
        self.relu=nn.ReLU()
    def forward(self,feature,x):
        b,c,h,w,z=x.shape

        feature = feature.unsqueeze(0).repeat(b, 1, 1)
        #print(feature.shape)
        x=x.view(b,h*w*z,c)
        shortcut=x
        x_csubject,_=self.attention(feature,x)
        #weights = torch.cosine_similarity(x, x_csubject, dim=2, eps=1e-08)
        #weights = self.relu(weights)
        #print(torch.max(weights))
        # print(torch.max(weights),torch.min(weights))
        #x = shortcut + x_csubject * weights.unsqueeze(2)
        x=x_csubject*0.5+shortcut*0.5
        # attn_windows = self.norm3(attn_windows_csubject )
        x = self.norm2(x)
        x=x.view(b,c,h,w,z)
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
class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        #self.csubject_former=CsubjectFormer(n_stages, features_per_stage,strides)

    def forward(self, x,feature=None,usefeature=False):
        skips = self.encoder(x)
        #skips1=self.csubject_former(skips,feature)
        return self.decoder(skips,feature,usefeature)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        #self.csubject_former = CsubjectFormer(n_stages, features_per_stage, strides)

    def forward(self, x,feature=None,usefeature=False):
        skips = self.encoder(x)
        #skips1 = self.csubject_former(skips, feature)
        return self.decoder(skips,feature,usefeature)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))
