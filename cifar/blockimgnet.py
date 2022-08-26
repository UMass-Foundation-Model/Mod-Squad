import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from themoe import MoE
from ykmoe import MoEBlock
from functools import partial

transform = transforms.Compose(
    [transforms.Resize((112,112)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class MyData(datasets.ImageFolder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self.samples[:(len(self.samples) // 10)]

trainset = MyData(os.path.join('/nobackup/projects/ILSVRC2012', 'train'), transform=transform) # 12811
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=16)

D = 128
import timm.models.vision_transformer
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, moe=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # del self.pos_embed
        # self.pos_embed = 0

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.moe = moe
        depth = 1
        if self.moe:
            self.blocks = nn.Sequential(*[
                MoEBlock(
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                    cvloss=0.1, switchloss=0.01, zloss=0.1,
                    num_attn_experts=16, head_dim=D//8 * 2,
                    dim=D, num_heads=8, mlp_ratio=4, qkv_bias=True)
                for i in range(depth)])
            for blk in self.blocks: 
                blk.attn.q_proj.f_gate.data.fill_(0.00)

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear    )):
            module.weight.data.fill_(0.00)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        z_loss = torch.FloatTensor([0]).cuda()
        for blk in self.blocks:
            if self.moe:
                x, aux_loss = blk(x)
                z_loss = z_loss + aux_loss
            else:
                x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome, z_loss

    def forward(self, x):
        x, z_loss = self.forward_features(x)
        x = self.head(x)
        return x, z_loss 

def vit_base(**kwargs):
    model = VisionTransformer(img_size=112,
        patch_size=16, embed_dim=D, depth=1, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_moe(**kwargs):
    model = VisionTransformer(
        img_size=112,
        moe=True,
        patch_size=16, embed_dim=D, depth=1, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_wide(**kwargs):
    model = VisionTransformer(img_size=112,
        patch_size=16, embed_dim=D*2, depth=1, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# choose = 0 # MoE 
# choose = 1 # Normal 
choose = 2 # wide
aux_weight = 1.0

print('D: ', D)
if choose == 0:
    net = vit_moe()
    print('ViT MoE!')
elif choose==1:
    net = vit_base()
    print('ViT Base!')
elif choose == 2:
    net = vit_wide()
    print('ViT Wide!')

net = nn.DataParallel(net).cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net.train()
for epoch in range(60):  # loop over the datase multiple times

    running_loss = 0.0
    aux_all = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, aux_loss = net(inputs)
        loss = criterion(outputs, labels)
        total_loss = loss + aux_loss * aux_weight
        total_loss.sum().backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if torch.is_tensor(aux_loss):
            aux_all += aux_loss.sum().item()

        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f z_loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / (i+1), aux_all / (i+1)))
            # running_loss = 0.0
            # aux_all = 0.0

print('Finished Training')
