import numpy as np

x=1
y=2
x+=y
print(x)
all_area=[]
for i in range(6):
    column_area = []
    for j in range(3):
        target_area = j
        column_area.append(target_area)
    all_area.append(column_area)

print(np.asarray(all_area))