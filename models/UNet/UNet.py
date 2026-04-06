
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU, Sequential, BatchNorm2d, MaxPool2d, Conv2d, ConvTranspose2d, Upsample
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torchvision.transforms
import torch


class conv_block(Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = Sequential(
            Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            BatchNorm2d(ch_out),
            ReLU(inplace=True),
            Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            BatchNorm2d(ch_out),
            ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x



class up_conv(Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = Sequential(
            Upsample(scale_factor=2),
            Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    BatchNorm2d(ch_out),
		    ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UNet(Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(UNet,self).__init__()
        
        self.Maxpool = MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



"""
class UNet(Module):

    def __init__(self, in_c, out_c):
        super(UNet, self).__init__()

        first_features = 32 

        #Encoder
        self.encoder_1 = UNet.block(in_c, first_features)
        self.pool1 = MaxPool2d(2,2)
        self.encoder_2 = UNet.block(first_features, first_features*2)
        self.pool2 = MaxPool2d(2,2)
        self.encoder_3 = UNet.block(first_features*2, first_features*4)
        self.pool3 = MaxPool2d(2,2)
        self.encoder_4 = UNet.block(first_features*4, first_features*8)
        self.pool4 = MaxPool2d(2,2)

        self.intermediate = UNet.block(first_features*8, first_features*16)

        # Decoder

        self.upsampconv_4 = ConvTranspose2d(first_features*16, first_features*8, kernel_size=2, stride=2)
        self.decoder_4 = UNet.block((first_features*8)*2, first_features*8)
        self.upsampconv_3 = ConvTranspose2d(first_features*8, first_features*4, kernel_size=2, stride=2)
        self.decoder_3 = UNet.block((first_features*4)*2, first_features*4)
        self.upsampconv_2 = ConvTranspose2d(first_features*4, first_features*2, kernel_size=2, stride=2)
        self.decoder_2 = UNet.block((first_features*2)*2, first_features*2)
        self.upsampconv_1 = ConvTranspose2d(first_features*2, first_features, kernel_size=2, stride=2)
        self.decoder_1 = UNet.block(first_features*2, first_features)   

        # Last Concolution
        self.final_conv = Conv2d(first_features, out_c, kernel_size=1)

    def forward(self,x):

        #Downsampling Part
        enc_1_output = self.encoder_1(x)
        enc_2_output = self.encoder_2(self.pool1(enc_1_output))
        enc_3_output = self.encoder_3(self.pool2(enc_2_output))
        enc_4_output = self.encoder_4(self.pool3(enc_3_output))
        
        intermediate = self.intermediate(self.pool4(enc_4_output))

        #Upsampling Part
        dec_4_output = self.upsampconv_4(intermediate)
        dec_4_output = UNet.crop_and_concat(enc_4_output, dec_4_output)
        dec_4_output = self.decoder_4(dec_4_output)

        dec_3_output = self.upsampconv_3(dec_4_output)
        dec_3_output = UNet.crop_and_concat(enc_3_output, dec_3_output)
        dec_3_output = self.decoder_3(dec_3_output)

        dec_2_output = self.upsampconv_2(dec_3_output)
        dec_2_output = UNet.crop_and_concat(enc_2_output, dec_2_output)
        dec_2_output = self.decoder_2(dec_2_output)

        dec_1_output = self.upsampconv_1(dec_2_output)
        dec_1_output = UNet.crop_and_concat(enc_1_output, dec_1_output)
        dec_1_output = self.decoder_1(dec_1_output)

        return torch.sigmoid(self.final_conv(dec_1_output))

    @staticmethod
    def block(in_c, out_c):
        return Sequential(
            Conv2d(in_c,out_c,kernel_size=3, padding=1, bias = False),
            BatchNorm2d(out_c),
            ReLU(inplace=True),
            Conv2d(out_c,out_c, kernel_size=3, padding=1, bias = False),
            BatchNorm2d(out_c),
            ReLU(inplace=True)
        )
    
    def crop_and_concat(enc_features, dec_features):
        _, _, H, W = dec_features.size()
        enc_features = torchvision.transforms.CenterCrop([H, W])(enc_features)
        return torch.cat((dec_features, enc_features), dim=1)

"""