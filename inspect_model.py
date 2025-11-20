# Inspect a PyTorch .pt model file
import torch

# Change this path to your model file if needed
model_path = "accident_best.pt"

model = torch.load(model_path, map_location='cpu')
print(model)

# If you want to see more details, uncomment below:
# if hasattr(model, 'state_dict'):
#     print(model.state_dict().keys())
# You can also use: print(model.__class__)
