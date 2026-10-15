import torch 
if torch.backends.mps.is_available(): 
    device = torch.device("mps") 
    print("Using Apple Silicon GPU") 
    
else: 
    device = torch.device("cpu")
    print("MPS not available, using CPU")
