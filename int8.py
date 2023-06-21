import torch

a = torch.randn(5)
print(a)

# 1a. find nomalzation constant
a_max = torch.abs(a).max()

# 1b. scale into range [-127, 127]
scaled = a / a_max*127

# 2. round to nearest value 
int8val = torch.round(scaled)

print(int8val)

# 3. dequantization by rescaling

fp32val = (int8val.float()*a_max/127.0)

print(f"value of random a = {a}")
print(f"scaled and rounded value of a = {int8val}")
print(f"dequant value: {fp32val}")
print(f"loss/noise: {torch.abs(fp32val - a).mean()}")
