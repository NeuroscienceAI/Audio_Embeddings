import torch
import torch.nn as nn
import json
from torchaudio.models import wav2vec2_model
import torchvision
# from torchvision.models import Swin_S_Weights
from torchvision import transforms
import torchaudio
from transformers import ClapModel, ClapProcessor
import numpy as np

class AlexNetClassifier(nn.Module):
    def __init__(self, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        self.alexnet = torchvision.models.alexnet(weights=weights if pretrained else None)
        # Modify the classifier to remove the last layer
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)===batch, 3, n-mels, length; RGB are the same
        
        # Define a transform pipeline that only resizes and crops the images
        # transform = transforms.Compose([
            # transforms.Resize(256),            # Resize the smaller edge to 256 pixels
            # transforms.CenterCrop(224),        # Crop the center 224x224 pixels
            # Note: Removed transforms.Normalize
        # ])

        # Apply the transformations and custom normalization
        # Assuming x is a batch of images with shape [32, 3, H, W] where H and W vary

        # Convert the batch of tensors to a list of individual tensors
        # x_list = torch.unbind(x, dim=0)
        
        # Apply the transform to each tensor in the list
        # x = torch.stack([transform(image) for image in x_list], dim=0)
        
        x /= x.max()            # normalize to [0, 1]; torch format
        # print('input spectrogram size=', x.shape) 
        ### dogs==[32batch, 3RGB, 128mels, 1000dimensions]
        ### If resize, All dataset is [32, 3, 224, 224]
        x = self.alexnet(x)
        ### dogs==torch.Size([32batch, 1000dimensions])
        ### If resize and remove one layer, ALL datasets=torch.Size([32, 4096])==[batch, 4096 dimensions]
        # print('extracted feature size=', x.shape) 
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, x
class ConvnextClassifier(nn.Module):
    def __init__(self, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        self.convnext = torchvision.models.convnext_base(weights=weights if pretrained else None)
        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)===batch, 3, n-mels, length; RGB are the same     
        # x /= x.max()            # normalize to [0, 1]; torch format
        ### All dataset will be modified to [32, 3, 232, 232] ##Convnext need 232 (different from others)
        # x = torchvision.transforms.Resize((232, 232))(x)
        # print('input spectrogram size=', x.shape) 
        ### dogs==[32batch, 3RGB, 128mels, 1000dimensions]
        
        ### Define a transform pipeline that only resizes and crops the images
        # transform = transforms.Compose([
        #     transforms.Resize(232),            # Resize the smaller edge to 232 pixels
        #     transforms.CenterCrop(224),        # Crop the center 224x224 pixels
            ### Note: Removed transforms.Normalize
        # ])

        ### Apply the transformations and custom normalization
        ### Assuming x is a batch of images with shape [32, 3, H, W] where H and W vary
        ### Convert the batch of tensors to a list of individual tensors
        # x_list = torch.unbind(x, dim=0)
        ### Apply the transform to each tensor in the list
        # x = torch.stack([transform(image) for image in x_list], dim=0)
        
        x /= x.max()            # normalize to [0, 1]; torch format
        x = self.convnext(x)
        
        ### dogs==torch.Size([32batch, 1000dimensions]) 
        # print('extracted feature size=', x.shape) 
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, x
class SwinClassifier(nn.Module):
    def __init__(self, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        weights = torchvision.models.Swin_S_Weights.DEFAULT
        self.swin = torchvision.models.swin_s(weights=weights if pretrained else None)
        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)===batch, 3, n-mels, length; RGB are the same    
        
        ### method 1>>> My own Normalize
        x /= x.max()            # normalize to [0, 1]; torch format
        
        ### method 2>>> Resize+CenterCrop+Normalize
        # weights = Swin_S_Weights.DEFAULT
        # preprocess = weights.transforms()
        # x = preprocess(x)
        
        ### method 3>>> Resize+CenterCrop+my own Normalize
        # transform = transforms.Compose([
        #     transforms.Resize(238),            # Resize the smaller edge to 238 pixels
        #     transforms.CenterCrop(224),        # Crop the center 224x224 pixels
        # ])
        # x_list = torch.unbind(x, dim=0)
        # x = torch.stack([transform(image) for image in x_list], dim=0)
        # x /= x.max()            # normalize to [0, 1]; torch format
  
        x = self.swin(x)
        
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, x
class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)===batch, 3, n-mels, length; RGB are the same
        x /= x.max()            # normalize to [0, 1]; torch format
        # print('input spectrogram size=', x.shape) ### dogs/gib==[32, 3, 128, 400], wat=[... 300], bat=[... 500]
        
        x = self.resnet(x)
        # print('extracted feature size=', x.shape) ### ALL datasets=torch.Size([32, 1000])==[batch, 1000 dimensions]
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, x


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0] # inputs x==[32batches, 10, 1, 96, 64]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        # print('vggish inputs shape=', x.shape) # dog data==[32*10, 1, 96, 64]
        out = self.vggish(x)
        # print('out1_shape=',out.shape) ###dogs==[32batch*10second, 128n-mels]
        out = out.reshape(batch_size, -1, out.shape[1]) 
        # print('out2_shape=',out.shape) ###dogs==[32, 10, 128]
        outs = out.mean(dim=1) ###average over time so all dataset will be the same (like AVES)
        # print('extracted feature size=', outs.shape) ### torch.Size([32, 128])==[batch size, in_features]
        # print(outs) # tensor=[batch size, 128n-mels] in the CUDA for both classification & detection
        logits = self.linear(outs) ### classification using "outs" as the embeddings

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, outs


class AvesClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes, embeddings_dim=768, multi_label=False):

        super().__init__()
        
        config_path='scripts/aves-base-bio.torchaudio.model_config.json'
        model_path='scripts/aves-base-bio.torchaudio.pt'
        
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)
        self.sample_rate = sample_rate
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    def load_config(self, config_path):
        # print('config_path=', config_path)
        # Check if config_path is a tuple and convert to string
        # if isinstance(config_path, tuple):
        #     config_path = config_path[0]  # Assuming it always contains at least one element
        #     print(f"config_path= {config_path}")
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj
    
    def forward(self, x, y=None):
        ### raw waveform without resampling###
        # target_sample_rate = 16000 # values from AVES paper
        # waveform = x.cpu() # detach data from GPU to CPU
        # if self.sample_rate != target_sample_rate:
        #     transform = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)
        #     waveform = transform(waveform) # still tensor format
        # waveform = waveform.to(1) # go back to GPU 0 or 1
        # out = self.model.extract_features(waveform)[0][-1]
        
        ###extract_feature in Torchaudio version will output all 12 layers' output, -1 to select the final layer
        out = self.model.extract_features(x)[0][-1]
        ###embedding for classification
        out = out.mean(dim=1)    # mean pooling over time like VGGish so all dataset will be the same
        logits = self.head(out)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, out
class AvesAllClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes, embeddings_dim=768, multi_label=False):

        super().__init__()
        
        config_path='scripts/aves-base-all.torchaudio.model_config.json'
        model_path='scripts/aves-base-all.torchaudio.pt'
        
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)
        self.sample_rate = sample_rate
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj
    
    def forward(self, x, y=None):
        out = self.model.extract_features(x)[0][-1]
        ###embedding for classification
        out = out.mean(dim=1)    # mean pooling over time like VGGish so all dataset will be the same
        logits = self.head(out)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits, out
    
class BioLingualClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        ### Load the BioLingual model online from HuggingFace
        # self.model_name = "davidrrobinson/BioLingual"
        # self.model = ClapModel.from_pretrained(self.model_name).to(0)
        # self.processor = ClapProcessor.from_pretrained(self.model_name)
        ### Load the CLAP based BioLingual model locally
        model_directory = "./BioLingual"
        self.model = ClapModel.from_pretrained(model_directory).to(1)
        self.processor = ClapProcessor.from_pretrained(model_directory)
 
        ### Output 512 features from CLAP/BioLingual model as the input features
        embeddings_dim=512
        self.linear = nn.Linear(in_features=embeddings_dim, out_features=num_classes)
        self.sample_rate = sample_rate
        self.multi_label = multi_label
        
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # FFT_SIZE_IN_SECS = 0.025
        # HOP_LENGTH_IN_SECS = 0.01
        n_mels = 128
        target_sample_rate = 48000
        # n_fft = int(FFT_SIZE_IN_SECS * self.sample_rate)
        # hop_length = int(HOP_LENGTH_IN_SECS * self.sample_rate)
        
        n_fft = 1024
        hop_length = 480
        # print('dataset sample rate=',self.sample_rate)
        waveform = x.cpu() # detach data from GPU to CPU
        ### Process the input audio array with the CLAP/BioLingual processor
        ### read tensor inputs, change to numpy, get tensor inside, and assign to GPU
        # if self.sample_rate != target_sample_rate:
        #     transform = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)
        #     waveform = transform(waveform) # still tensor format
        waveform_np = waveform.numpy()  # Convert tensor (in CPU) to NumPy array
        ### in demo code, inputs are raw audio waveform in NumPy
        inputs = self.processor(audios = waveform_np, sampling_rate=target_sample_rate, 
                                feature_size=n_mels, fft_window_size=n_fft,
                                hop_length=hop_length, n_fft=n_fft, return_tensors="pt").to(1)
        
        # inputs = self.processor(audios = waveform_np, sampling_rate=target_sample_rate, 
        #                        feature_size=n_mels, fft_window_size=n_fft,
        #                        hop_length=hop_length, n_fft=n_fft, return_tensors="pt")
        # Move processed inputs to the same device as model
        # Code from GPT4, seems useless
        # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get audio features
        with torch.no_grad():
            audio_embed = self.model.get_audio_features(**inputs)
            ### Embedded feature is 512 and same for data/real/H/T
            ### I examined this through their provided code
            audio_features = audio_embed.data # tensor (1, 512)
        
        # Pass the features through a linear layer to get logits for classification
        logits = self.linear(audio_features)
        
        ### for rfcx=24types and batch=4, logits==
        ### 4 tensors, and each tensor has 24 values [-0.0425, ..., 0.0260]
        # print('logits=', logits)
        
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        
        return loss, logits, audio_features

class ClapClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

         ### Load the CLAP model online from HuggingFace
         # self.model_name = "laion/clap-htsat-unfused"
         # self.model = ClapModel.from_pretrained(self.model_name).to(0)
         # self.processor = ClapProcessor.from_pretrained(self.model_name)
        ### Load the CLAP model locally
        model_directory = "./CLAP_LAION"
        self.model = ClapModel.from_pretrained(model_directory).to(0)
        self.processor = ClapProcessor.from_pretrained(model_directory)
 
        ### Output 512 features from CLAP model as the input features
        embeddings_dim=512
        self.linear = nn.Linear(in_features=embeddings_dim, out_features=num_classes)
        self.sample_rate = sample_rate
        self.multi_label = multi_label
        
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        FFT_SIZE_IN_SECS = 0.05
        HOP_LENGTH_IN_SECS = 0.01
        n_mels = 128
        target_sample_rate = 48000
        n_fft = int(FFT_SIZE_IN_SECS * self.sample_rate)
        hop_length = int(HOP_LENGTH_IN_SECS * self.sample_rate)
        # Process the input audio array with the CLAP processor
        
        ### below line from Beans will squeeze all batches to one and wrong!
        # waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        ### match all sample rates to desired 48kHz
        waveform = x.cpu() # detach data from GPU to CPU
        if self.sample_rate != target_sample_rate:
            transform = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)
            waveform = transform(waveform) # still tensor format
        waveform_np = waveform.numpy()  # Convert tensor (in CPU) to NumPy array
        waveform_np = waveform_np / np.abs(waveform_np).max()
        print('waveform_np_max=', np.abs(waveform_np).max())
        ### same size as input waveforms
        # print('waveform_np_size', waveform_np.shape)
        ### in demo code, inputs are raw audio waveform in NumPy
        inputs = self.processor(audios = waveform_np, sampling_rate=target_sample_rate, 
                                feature_size=n_mels, fft_window_size=n_fft,
                                hop_length=hop_length, n_fft=n_fft, return_tensors="pt", ).to(0)
        print(inputs.input_features)
        # Get audio features
        with torch.no_grad():
            audio_embed = self.model.get_audio_features(**inputs)
            audio_features = audio_embed.data # tensor (1, 512)
        
        # Pass the features through a linear layer to get logits for classification
        ### [batch_size, 512]
        # print('audio_features_size', audio_features.shape)
        logits = self.linear(audio_features)
        ### [batch_size, num of types]
        # print('logits=', logits)
        
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)
        
        return loss, logits, audio_features