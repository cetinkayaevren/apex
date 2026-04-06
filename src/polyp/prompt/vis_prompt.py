from torch.nn import Module
from torch.nn import Parameter
from torch.nn import functional as F
from torch import fft
import torch

generator1 = torch.Generator().manual_seed(42)

class VisualPrompt(Module):

    def __init__(self, prompt_dim=6, img_shape=352, prompt=None):
        super(VisualPrompt, self).__init__()
        self.prompt_dim = prompt_dim
        self.img_shape = img_shape
        self.padding_size = (self.img_shape - self.prompt_dim)//2
        self.__init = 0.03
        if prompt==None:
            self.prompt = Parameter(torch.randn(1, 3, self.prompt_dim, self.prompt_dim)*self.__init) #Parameter(torch.ones(1, 3, self.prompt_dim, self.prompt_dim),requires_grad=True) #Parameter(torch.randn(1, 3, self.prompt_dim, self.prompt_dim,generator=generator1)) #Parameter(torch.ones(1, 3, self.prompt_dim, self.prompt_dim),requires_grad=True) #Learnable Prompt Intialization
        else:
            self.prompt = prompt

    def update_prompt(self, new_data):
        with torch.no_grad():
            self.prompt.copy_(new_data)

    def forward(self, x):
        _,_, height, width = x.size()

        x_copy = x.clone()

        fourier_domain = fft.fft2(x_copy) #Discrete Fourier Transform, take the last two dimension

        #Get amplitude and phase information
        amplitude_domain = torch.abs(fourier_domain)
        frequency_domain = torch.angle(fourier_domain)

        amplitude_domain_centered = fft.fftshift(amplitude_domain) #Get centered amplitude domain

        #Prompt Padding
        padded_prompt = F.pad(self.prompt, 
                              [self.padding_size, self.padding_size, self.padding_size, self.padding_size],
                              mode="constant",
                              value=1.0).contiguous()
        
        #Concatenate Amplitude and Prompt
        concat_prompt_amp = amplitude_domain_centered * padded_prompt

        #Reshift the amplictude image concatenated by prompt
        concat_prompt_amp = fft.ifftshift(concat_prompt_amp)

        # Recompose fft

        real = torch.cos(frequency_domain) * concat_prompt_amp
        imag = torch.sin(frequency_domain) * concat_prompt_amp
        fft_src_ = torch.complex(real=real, imag=imag)

        prompted_image = fft.ifft2(fft_src_, dim=(-2, -1), s=[height, width]).real

        return prompted_image