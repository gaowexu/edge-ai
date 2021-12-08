from dlr.counter.phone_home import PhoneHome
PhoneHome.disable_feature()
import dlr
import numpy as np
import time


# Load model.
# /path/to/model is a directory containing the compiled model artifacts (.so, .params, .json)
model = dlr.DLRModel('./clsmodel', 'gpu', 0)

# Prepare some input data.
x = np.random.rand(1, 3, 224, 224)

# warm up
for _ in range(5):
    y = model.run(x)


# estimation
t1 = time.time()
iter_times = 100
for _ in range(iter_times):
    y = model.run(x)
t2 = time.time()

print("y = {}".format(y))
print("Time cost for each frame (size=1x3x224x224) = {} ms".format(1000 * (t2 - t1) / iter_times))

