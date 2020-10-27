import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_scatter import scatter_max

class LPMNet(nn.Module):
    def __init__(self, n_points, latent_size, part_count):
        super(LPMNet, self).__init__()
        
        self.latent_size = latent_size
        self.n_points = n_points
        self.part_count = part_count
        
        #Encoder
        self.enc_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.enc_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.enc_conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.enc_bn1 = nn.BatchNorm1d(64)
        self.enc_bn2 = nn.BatchNorm1d(128)

        
        #Segmenter
        self.seg_conv1 = torch.nn.Conv1d(self.latent_size*2, self.latent_size, 1)
        self.seg_conv2 = torch.nn.Conv1d(self.latent_size, self.latent_size//2, 1)
        self.seg_conv3 = torch.nn.Conv1d(self.latent_size//2, self.latent_size//4, 1)
        self.seg_conv4 = torch.nn.Conv1d(self.latent_size//4, self.part_count+1, 1)
        self.seg_bn1 = nn.BatchNorm1d(self.latent_size)
        self.seg_bn2 = nn.BatchNorm1d(self.latent_size//2)
        self.seg_bn3 = nn.BatchNorm1d(self.latent_size//4)
        
        #Decoder
        self.dec1 = nn.Linear(self.latent_size,self.n_points//2)
        self.dec2 = nn.Linear(self.n_points//2,self.n_points)
        self.dec3 = nn.Linear(self.n_points,self.n_points*3)
        
        
    def get_point_features(self,point_cloud):
        points = point_cloud[:,:,0:3]
        points = points.permute(0,2,1).contiguous()
        
        x = F.relu(self.enc_bn1(self.enc_conv1(points)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_conv3(x)
        return x.permute(0,2,1).contiguous()
    
    def segment(self, point_features):  
        global_code = torch.max(point_features, 1)[0]
        global_code = global_code.unsqueeze(1).repeat(1,point_features.shape[1],1)
        seg_features = torch.cat((point_features, global_code),2)
        
        x = F.relu(self.seg_bn1(self.seg_conv1(seg_features.permute(0,2,1).contiguous())))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)
        return x.permute(0,2,1).contiguous()
        
        
    def slice_tensor(self,tensor,indices,number):
        mask = (indices == number)
        masked = (tensor * mask.unsqueeze(2).repeat(1,1,self.latent_size))
        maxed = torch.max(masked, 1, keepdim=True)[0]
        return maxed
    
    def get_part_features(self, point_features, seg_results):
        labels = seg_results.argmax(dim=2,keepdim=True).squeeze()
        
        part_codes = []
        for indice in range(self.part_count):
            part_codes.append(self.slice_tensor(point_features,labels,indice+1))
            
        all_indices = torch.cat(part_codes,dim=1)
        return all_indices   
    
    #def get_part_features(self, point_features, seg_results):
    #    labels = seg_results.argmax(dim=2,keepdim=True).squeeze()
    #    part_codes, a_max = scatter_max(point_features,labels.detach().long(), dim=1)
    #    all_indices = part_codes[:,1:,:]
    #    return all_indices   

    def decode(self, global_code):
        x = F.relu(self.dec1(global_code))
        x = F.relu(self.dec2(x))
        x = torch.tanh(self.dec3(x))
        return x.view(-1, self.n_points, 3)
    
    def forward(self,x):
        
        point_features = self.get_point_features(x)
        seg_results = self.segment(point_features)
        part_features = self.get_part_features(point_features, seg_results)
        global_code = torch.max(part_features, 1)[0]
        reconstructed = self.decode(global_code)
        
        return seg_results, reconstructed