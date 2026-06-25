import torch 
# Check if MPS (Metal Performance Shaders) is available 
if torch.backends.mps.is_available(): 
    device = torch.device("mps") 
    print("Using Apple Silicon GPU") 
    
else: 
    device = torch.device("cpu")
    print("MPS not available, using CPU") # Example: Move a tensor to your M4 GPU x = torch.rand(5, 3).to(device) print(x.device)