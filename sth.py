import numpy as np
a = np.arange(0, 5, 1)
b = np.arange(5, 10, 1)
nums = [a, b]
ans = []
for i in range(2):
    ans.append(nums[i])
ans = np.array(ans)
print(ans)