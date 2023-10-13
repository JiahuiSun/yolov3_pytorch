from torch import nn
import torch


class Conv2dUnit(nn.Module): # Conv + BN + LeakyReLU = CBL
    def __init__(self, input_dim, filters, kernels, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, filters, kernel_size=kernels, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.LeakyReLU(x)
        return x

# 3, 2, (3, 3), stride=1, padding=1
class domainAttention(nn.Module): # N,1,1,2C -> N,1,1,2C/S 
    def __init__(self, input_dim, output_dim):
        super(domainAttention, self).__init__()
        self.input_dim1 = input_dim
        self.output_dim1 = output_dim
        self.input_dim2 = self.output_dim1
        self.output_dim2 = 2

        self.fc1 = nn.Linear(self.input_dim1,self.output_dim1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.input_dim2,self.output_dim2,bias=False) # N,1,1,2

        self.sigmoid = nn.Softmax(dim=1)
        
    def forward(self,x):

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class channelAttention(nn.Module):
    def __init__(self,input_channel,batch_size):
        super(channelAttention, self).__init__()
        self.batch_size = batch_size
        self.input_channel = input_channel
        self.avg_input1 = input_channel
        self.avg_output1 = int(input_channel/2)
        self.avg_input2 = self.avg_output1
        self.avg_output2 = input_channel # self.avg_input2

        self.max_input1 = input_channel
        self.max_output1 = int(input_channel/2)
        self.max_input2 = self.max_output1
        self.max_output2 = input_channel # self.max_input2

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # N,C,H,W -> N,C,1,1 / C,H,W -> C,1,1   N,3,320,160 ->N,3,1,1
        self.max_pool = nn.AdaptiveMaxPool2d(1) 

        self.avg_fc1   = nn.Linear(self.avg_input1,self.avg_output1,bias=False)
        self.avg_relu1 = nn.ReLU()
        self.avg_fc2   = nn.Linear(self.avg_input2,self.avg_output2,bias=False)
        self.avg_sigmoid = nn.Sigmoid()

        self.max_fc1   = nn.Linear(self.max_input1,self.max_output1, bias=False)
        self.max_relu1 = nn.ReLU()
        self.max_fc2   = nn.Linear(self.max_input2,self.max_output2, bias=False)
        self.max_sigmoid = nn.Sigmoid()

        self.domainAttention = domainAttention(input_channel*2,input_channel) # 2C = 6



    def forward(self,x):


        avg_pool = self.avg_pool(x)
        avg_pool = avg_pool.view(self.batch_size,1,1,self.input_channel) 

        
        max_pool = self.max_pool(x).view(self.batch_size,1,1,self.input_channel) # turn (3,1,1) -> (1,1,3)

        avg_x = self.avg_fc1(avg_pool)
        avg_x = self.avg_relu1(avg_x)
        avg_x = self.avg_fc2(avg_x)
        avg_x = self.avg_sigmoid(avg_x)
        max_x = self.max_fc1(max_pool)
        max_x = self.max_relu1(max_x)
        max_x = self.max_fc2(max_x)
        max_x = self.max_sigmoid(max_x)

    
        raw_attention = torch.cat([avg_x,max_x],dim=3) # if input dim = 4 ,dim =3;input dim = 3,dim = 2
        attention = self.domainAttention(raw_attention)


        weight = torch.cat([avg_x,max_x],dim=2) # if input dim = 4 ,dim =2;input dim = 3,dim = 1 
        x = torch.matmul(attention, weight)

        return x


class AttentionConv2dUnit(nn.Module): # Conv2d + BN + LeakyReLU + attention
    def __init__(self, input_dim, filters, kernels, stride, padding,batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.CBL = Conv2dUnit(input_dim, filters, kernels, stride, padding)
        self.attention = channelAttention(filters,self.batch_size) # channel attention unit
        self.attention_input_channel = filters
    def forward(self,x):
        
        cbl = self.CBL(x)
        x = self.attention(cbl)
        x = x.view(self.batch_size,self.attention_input_channel,1,1) * cbl # element-wise multiplication on each channel dim 
        return x
            




class ResidualBlock(nn.Module):
    def __init__(self, input_dim, filters):
        super().__init__()
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x

class spacialAttention(nn.Module): # SA
    def __init__(self,input_dim):
        super().__init__()
        self.compressRate = 2
        self.stride = 1
        self.padding = 1
        self.conv1 = nn.Conv2d(input_dim,int(input_dim/self.compressRate), kernel_size=(1,1), stride=self.stride, padding=0, bias=False)
        self.conv2 = nn.Conv2d(int(input_dim/self.compressRate), 1, kernel_size=(3,3), stride=self.stride, padding=self.padding, bias=False)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class CAResidualBlock(nn.Module): # CA + ResidualBlock
    def __init__(self, input_dim, filters,batch_size): 
        super().__init__()
        self.batch_size = batch_size
        self.input_channel = input_dim
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)
        self.attention_input_channel = 2*filters#attention_input_channel
        self.channelAttention = channelAttention(2*filters,batch_size)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        cbl = self.conv2(x)
        x = self.channelAttention(cbl)
        x = x.view(self.batch_size,self.input_channel,1,1) * cbl # element-wise multiplication on each channel dim 
        x += residual
        return x

class SAResidualBlock(nn.Module): # SA + ResidualBlock
    def __init__(self, input_dim, filters): 
        super().__init__()
        
        self.input_channel = input_dim
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)
        self.spacialAttention = spacialAttention(2*filters)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        cbl = self.conv2(x)
        x = self.spacialAttention(cbl)

        x = x * cbl # element-wise multiplication on each channel dim  # .view(1,self.input_channel,1,1)
        x += residual
        return x

class StackResidualBlock(nn.Module):
    def __init__(self, input_dim, filters, n):
        super().__init__()
        self.resx = nn.Sequential()
        for i in range(n):
            self.resx.add_module('stack_%d' % (i+1,), ResidualBlock(input_dim, filters))

    def forward(self, x):
        return self.resx(x)

class AttentionStackResidualBlock(nn.Module):
    def __init__(self, input_dim, filters, n,batch_size):
        super().__init__()
        self.resx = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.resx.add_module('stack_%d' % (i+1,), SAResidualBlock(input_dim, filters))
            else:
                self.resx.add_module('stack_%d' % (i+1,), CAResidualBlock(input_dim, filters,batch_size))
            

    def forward(self, x):
        return self.resx(x)


class Darknet(nn.Module):
    def __init__(self, initial_filters=32,batch_size=1):
        super().__init__()
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        # darknet53所有卷积层都没有偏移，bias=False
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)
        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)
        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        #self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)
        self.stack_residual_block_3 = AttentionStackResidualBlock(i256, i128, n=8,batch_size=batch_size)
        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        #self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)
        self.stack_residual_block_4 = AttentionStackResidualBlock(i512, i256, n=8,batch_size=batch_size)
        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
        # self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)
        self.stack_residual_block_5 = AttentionStackResidualBlock(i1024, i512, n=4,batch_size=batch_size)

        self.CBL5_1 = nn.Sequential(
            AttentionConv2dUnit(i1024, i512, (1, 1), stride=1, padding=0,batch_size=batch_size),
            AttentionConv2dUnit(i512, i1024, (3, 3), stride=1, padding=1,batch_size=batch_size),
            AttentionConv2dUnit(i1024, i512, (1, 1), stride=1, padding=0,batch_size=batch_size),
            AttentionConv2dUnit(i512, i1024, (3, 3), stride=1, padding=1,batch_size=batch_size),
            AttentionConv2dUnit(i1024, i512, (1, 1), stride=1, padding=0,batch_size=batch_size)
        )
        """
        # FPN+YOLO head
        self.CBL5_1 = nn.Sequential(
            Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0),
            Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1),
            Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0),
            Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1),
            Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        )
        """
        self.yolo_head1 = nn.Sequential(
            Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1),
            nn.Conv2d(i1024, 3*5, kernel_size=(1, 1))
        )

        self.conv7 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        """
        self.CBL5_2 = nn.Sequential(
            Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0),
            Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1),
            Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0),
            Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1),
            Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        )
        """
        self.CBL5_2 = nn.Sequential(
            AttentionConv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0,batch_size=batch_size),
            AttentionConv2dUnit(i256, i512, (3, 3), stride=1, padding=1,batch_size=batch_size),
            AttentionConv2dUnit(i512, i256, (1, 1), stride=1, padding=0,batch_size=batch_size),
            AttentionConv2dUnit(i256, i512, (3, 3), stride=1, padding=1,batch_size=batch_size),
            AttentionConv2dUnit(i512, i256, (1, 1), stride=1, padding=0,batch_size=batch_size)
        )
        self.yolo_head2 = nn.Sequential(
            Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1),
            nn.Conv2d(i512, 3*5, kernel_size=(1, 1))
        )

        self.conv8 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        """
        self.CBL5_3 = nn.Sequential(
            Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0),
            Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1),
            Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0),
            Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1),
            Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        )
        """
        self.CBL5_3 = nn.Sequential(
            AttentionConv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0,batch_size=batch_size),
            AttentionConv2dUnit(i128, i256, (3, 3), stride=1, padding=1,batch_size=batch_size),
            AttentionConv2dUnit(i256, i128, (1, 1), stride=1, padding=0,batch_size=batch_size),
            AttentionConv2dUnit(i128, i256, (3, 3), stride=1, padding=1,batch_size=batch_size),
            AttentionConv2dUnit(i256, i128, (1, 1), stride=1, padding=0,batch_size=batch_size)
        )
        self.yolo_head3 = nn.Sequential(
            Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1),
            nn.Conv2d(i256, 3*5, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stack_residual_block_1(x)
        x = self.conv3(x)
        x = self.stack_residual_block_2(x)
        x = self.conv4(x)
        feat_downsample8 = self.stack_residual_block_3(x)
        x = self.conv5(feat_downsample8)
        feat_downsample16 = self.stack_residual_block_4(x)
        x = self.conv6(feat_downsample16)
        feat_downsample32 = self.stack_residual_block_5(x)

        downsample32 = self.CBL5_1(feat_downsample32)

        x = self.conv7(downsample32)
        x = self.upsample1(x)
        x = torch.cat((x, feat_downsample16), dim=1)
        downsample16 = self.CBL5_2(x)

        x = self.conv8(downsample16)
        x = self.upsample2(x)
        x = torch.cat((x, feat_downsample8), dim=1)
        downsample8 = self.CBL5_3(x)
        
        y3 = self.yolo_head3(downsample8)
        y3 = y3.view(y3.size(0), 3, 5, y3.size(2), y3.size(3))  # reshape

        y3 = y3.permute(0, 1, 3, 4, 2)
        return y3


class Darknet10(nn.Module):
    def __init__(self, initial_filters=32):
        super().__init__()
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32
    
        # darknet53所有卷积层都没有偏移，bias=False
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)

        # FPN+YOLO head
        self.conv7 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv8 = Conv2dUnit(i1024, i256, (1, 1), stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.yolo_head3 = nn.Sequential(
            Conv2dUnit(i512, i256, (3, 3), stride=1, padding=1),
            nn.Conv2d(i256, 3*5, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feat_downsample8 = self.conv4(x)
        feat_downsample16 = self.conv5(feat_downsample8)
        feat_downsample32 = self.conv6(feat_downsample16)

        x = self.conv7(feat_downsample32)
        x = self.upsample1(x)
        downsample16 = torch.cat((x, feat_downsample16), dim=1)

        x = self.conv8(downsample16)
        x = self.upsample2(x)
        downsample8 = torch.cat((x, feat_downsample8), dim=1)

        y3 = self.yolo_head3(downsample8)
        y3 = y3.view(y3.size(0), 3, 5, y3.size(2), y3.size(3))  # reshape

        y3 = y3.permute(0, 1, 3, 4, 2)
        return y3

if __name__ == '__main__':
    from torchsummary import summary
    
    # 验证darknet53
    model = Darknet(batch_size=5)
    if torch.cuda.is_available():
        model = model.cuda()
    input = torch.randn(5,3,160,320)
    if torch.cuda.is_available():
        input = input.cuda()
    model(input)


    """ 
    # 验证domainAttention模块
    da = domainAttention(6,3)
    if torch.cuda.is_available():
        da = da.cuda()
    input = torch.randn(1,1,6)
    if torch.cuda.is_available():
        input = input.cuda()
    da(input)
    """


    """ 
    # 验证channelAttention模块
    ca = channelAttention(3)
    if torch.cuda.is_available():
        ca = ca.cuda()
    input = torch.randn(3,320,160)
    if torch.cuda.is_available():
        input = input.cuda()
    ca(input)
    """ 

    """
    ac = AttentionConv2dUnit(3,32,(3,3),1,1,32) # input_dim, filters, kernels, stride, padding , channel
    if torch.cuda.is_available():
        ac = ac.cuda()
    input = torch.randn(1,3,320,160)
    if torch.cuda.is_available():
        input = input.cuda()
    ac(input)
    """
    
    """
    # 验证spacialAttention模块
    sa = spacialAttention(64)
    if torch.cuda.is_available():
        sa = sa.cuda()
    input = torch.randn(1,64,320,160)
    if torch.cuda.is_available():
        input = input.cuda()
    sa(input)
    """

