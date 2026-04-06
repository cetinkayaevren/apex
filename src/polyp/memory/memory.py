import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Linear, ReLU, Parameter, Flatten
from torch.nn import functional as F
from torch import fft
from cont_loss.losses import SupConLoss


class Memory(Module):
    def __init__(self, memory_size = 100, prompt_size = 6, temperature=0.1, cont_temp=0.07,  device = "cuda:0", lambda_factor=1):
        super(Memory, self).__init__()

        self.device = device
        self.cont_temp = cont_temp
        self.batch_size = 16
        #Supervised Contrastive Loss
        self.supcon_loss = SupConLoss(temperature=self.cont_temp, device=self.device)
        self.patch_size = prompt_size #for 6x6x3 prompts
        self.memory_size = memory_size
        self.temperature = temperature # For soft addressing
        self.in_features = self.patch_size*self.patch_size*3
        self.lambda_factor=lambda_factor
        
        self.latent_feature = 36
        self.contrastive_out= 36

        self.flatten = Flatten()
        
        self.motion_matching_encoder = Sequential(  
            Linear(in_features= self.in_features, out_features=72),
            ReLU(),

            Linear(in_features= 72, out_features= 72),
            ReLU(),

            Linear(in_features= 72, out_features= 36),
            ReLU(),

            Linear(in_features= 36, out_features= self.latent_feature)

        )

        self.decoder_model = Sequential(
            Linear(in_features= self.latent_feature, out_features=36),
            ReLU(),
            Linear(in_features= 36, out_features=72),
            ReLU(),
            Linear(in_features= 72, out_features=72),
            ReLU(),
            Linear(in_features=72, out_features= self.in_features)
        )

        self.contrastive_mlp = Sequential(
            Linear(in_features= 24, out_features=24),
            ReLU(),
            Linear(in_features= 24, out_features=36),
            ReLU(),
            Linear(in_features=36, out_features=36),
        )

        self.memory_shape = [self.memory_size, 24]
        self.memory_w = nn.init.normal_(torch.empty(self.memory_shape), mean=0.0, std=1.0)
        self.memory_w = Parameter(self.memory_w, requires_grad=True)

    def forward(self, input_img, domain_gt):
        motion_encoder =  self.motion_matching_encoder
        
        b_size, channel, height, width = input_img.size()[0], input_img.size()[1], input_img.size()[2], input_img.size()[3] 
        padding_size = (height - self.patch_size)//2

        half_patch = self.patch_size // 2

        center_x, center_y = height //2, width //2

        input_img_copy = input_img.clone()

        #Discrete Fourier Transform
        fourier_domain = fft.fft2(input_img_copy)
        fourier_domain_centered = fft.fftshift(fourier_domain)

        #Get amplitude and phase information
        amplitude_domain_centered = torch.abs(fourier_domain_centered)
        frequency_domain_centered = torch.angle(fourier_domain_centered)
        
        #Get Patch from the amplitude domain
        patch_amplitude_domain = amplitude_domain_centered[:, :, center_y-half_patch:center_y+half_patch, center_x-half_patch:center_x+half_patch]
        flattened_img = self.flatten(patch_amplitude_domain)
        
        #Encoder pass to get latent features of given flattened image
        frequency_feature = motion_encoder(flattened_img)  

        #Normalize
        frequency_feature_normalized = F.normalize(frequency_feature, dim=1)

        #Sent to Contrastive MLP for computing Contrastive Loss
        contrastive_feature = self.contrastive_mlp(frequency_feature_normalized)
        contrastive_feature = contrastive_feature.reshape(-1, 1, 1, self.contrastive_out)
        contrastive_loss = self.supcon_loss(contrastive_feature, domain_gt)

        #Normalize Memory
        memory_norm = F.normalize(self.memory_w, dim=1)

        s = torch.matmul(frequency_feature_normalized, memory_norm.T)

        # Soft Addressing for getting Addressing Vector
        addressing_vec = F.softmax(s/self.temperature, dim=-1)
        memory_feature = torch.matmul(addressing_vec, self.memory_w)
        
        # Reconstruct the Visual Prompt
        decoded_prompt = self.decoder_model(memory_feature)
        recons_prompt = decoded_prompt.reshape(b_size, channel, self.patch_size, self.patch_size)

        # Soften the Visual Prompt via Interpolation
        recons_prompt = ((1 - self.lambda_factor) + self.lambda_factor * recons_prompt)

        # Add 1s padding to the Prompt as the same size with the image
        padded_prompt = F.pad(recons_prompt, 
                              [padding_size, padding_size, padding_size, padding_size],
                              value=1.0).contiguous()
        
        #Concatenate Amplitude and Prompt
        concat_prompt_amp = amplitude_domain_centered * padded_prompt

        #Reshift the amplictude image concatenated by prompt
        concat_prompt_amp = fft.ifftshift(concat_prompt_amp)
        freq_domain_final = fft.ifftshift(frequency_domain_centered)

        #Inverse Fast Fourier to reconstruct the image
        real = torch.cos(freq_domain_final) * concat_prompt_amp
        imag = torch.sin(freq_domain_final) * concat_prompt_amp
        fft_src_ = torch.complex(real=real, imag=imag)
        prompted_image = fft.ifft2(fft_src_, dim=(-2, -1), s=[height, width]).real
        
        return prompted_image, addressing_vec, recons_prompt, contrastive_loss