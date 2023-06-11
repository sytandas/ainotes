inputs = [1.2, 3.2, 8.8]

weights1 = [0.8, 0.45, -0.3]
weights2 = [0.6, 0.98, 0.2]
weights3 = [0.82, 0.27, 0.11]


bias1 = 0.03 
bias2 = 0.4 
bias3 = 0.12 
bias4 = 2

# basic is (inputs * weights) + bias

output = (inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + bias1) 
+ (inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + bias2) 
+ (inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + bias3)
+ (inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + bias4)


print(output)
