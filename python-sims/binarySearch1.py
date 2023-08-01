from lpafunctions import *
import numpy as np
from sklearn.linear_model import LinearRegression

# test = [[0, 3]
#         [1, 2]
#         [2, 9]
#         [3, 10]]

# print(test)

# test2 = test[]



N_values = np.zeros(11)
for i in range(np.size(N_values)):
    N_values[i] = (1000 * int(np.power(2, i)))
N_values = N_values.astype(np.int64)

# N_values = [100, 200, 300, 400]
print("hello")
estThresholdDegrees, estThresholdVs = ER_BinSearchThreshold_v(32, N_values)
print("estThresholdDegrees:")
print(estThresholdDegrees)
print("estThresholdVs:")
print(estThresholdVs)

# get results
file_object = open('ERBinSearch_results_upto%d.txt' % N_values[-1], 'w')
x = np.log2(N_values)
y = np.log2(estThresholdDegrees)
model = LinearRegression().fit(np.reshape(x, (-1,1)), np.reshape(y, (-1,1)))
r_squared = model.score(np.reshape(x, (-1,1)), np.reshape(y, (-1,1)))
file_object.write("R squared error:\n")
file_object.write(str(r_squared))
file_object.write('\n')
y_intercept = model.intercept_
file_object.write("y-intercept:\n")
file_object.write(str(y_intercept))
file_object.write('\n')
slope = model.coef_
file_object.write("slope:\n")
file_object.write(str(slope[0]))
file_object.write('\n')
# plot threshold degrees against N wth best fit line
plt.scatter(x, y, c = "red")
# um = "log2(N*p) = %fN + %f" % (slope, y_intercept)
plt.plot(x, slope[0]*x + y_intercept)
plt.title('Estimated threshold (0.5 consensus) degree vs. N')
plt.xlabel('log(N)')
plt.ylabel('Log of estimated threshhold degree = log(N*p)')
plt.savefig('testplot')