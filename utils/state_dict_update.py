import torch 


model= torch.load('/data/FaderNetworks/models/male.pth')

torch.save(model.state_dict(), '/data/FaderNetworks/models/updated_male.pth')

print(model.state_dict())




