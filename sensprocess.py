import numpy as np
import plotly.graph_objects as go
import scipy.signal
# from scipy.signal import savgol_filter
import datetime

arr = np.genfromtxt('/home/michael/Documents/data/astro/032022/track2303/crab2303.csv').T


print(arr.shape)
tdiff = np.diff(arr[0]).mean()
print(tdiff)


fig = go.Figure()

# t = np.array((arr[0] + 3600*3) * 1000, dtype='datetime64[ms]')
# fig.add_trace(go.Scatter(y=arr[1]))
# fig.show()
# exit()

sb, se = 40000, 180000

t = np.array((arr[0, sb:se] + 3600*3) * 1000, dtype='datetime64[ms]')
y = arr[1, sb:se]

ip = scipy.signal.periodogram(y, fs=1/tdiff)
fig.add_trace(go.Scatter(x=ip[0], y=ip[1]))
fig.show()
exit()

p = np.fft.fft(y)
f = np.fft.fftfreq(len(p), d=tdiff)

p[np.abs(p)**2 > 616000000] = 0
p[np.abs(p)**2 < 40000000] = 0
p[np.abs(f) < 26] = 0

# fig.add_trace(go.Scatter(x=f, y=np.abs(p)**2))
# fig.show()
# exit()


ip = np.abs(np.fft.ifft(p))**2
fig.add_trace(go.Scatter(x=t, y=ip))

fig.show()

# def running_smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
#
# def avg_smooth(y, n_pts):
#     l = (len(y) // n_pts) * n_pts
#     a = np.array(y[0:l])
#     a = a.reshape(-1, n_pts)
#     return np.mean(a, axis=1)
#
#
# ip = scipy.signal.savgol_filter(ip, 8, 0)
