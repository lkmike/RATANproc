import sys
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import savgol_filter

dt = np.dtype([
    ('cnt', '<u4'),
    ('avg_kurt', '<u4'),
    ('state', '>u4'),
    ('channel', '>u4'),
    ('data', '<u8', 0x80)
])

block_array = np.fromfile("/home/michael/Documents/data/astro/022022/2022-01-27_11.bin", dtype=dt)
count = len(block_array)

ch0_max = 0
ch1_max = 0
for el in block_array:
    if el['channel'] == 0:
        if el['cnt'] > ch0_max:
            ch0_max = el['cnt']
    else:
        if el['cnt'] > ch1_max:
            ch1_max = el['cnt']

avg_num = 2 ** (block_array[0]['avg_kurt'] & 0b111111)
spectrum_length = 8192 // avg_num

ch0_length = ((ch0_max + 1) // 0x80) * 0x80
ch1_length = ((ch1_max + 1) // 0x80) * 0x80
ch_length = max(ch0_length, ch1_length)

chan0 = np.zeros(ch_length, dtype=dt)
chan1 = np.zeros(ch_length, dtype=dt)

for el in block_array:
    if (el['channel'] == 0) and (el['cnt'] < ch0_length):
        chan0[el['cnt']] = el
    elif el['cnt'] < ch1_length:
        chan1[el['cnt']] = el

del block_array

cc0 = chan0['data'].reshape(-1, spectrum_length)
ch0_data = (cc0 & 0x7FFFFFFFFFFFFF).astype(np.float32)
ch0_kurt = (cc0 >> 55).astype(np.float32)
del cc0

cc1 = chan1['data'].reshape(-1, spectrum_length)
ch1_data = (cc1 & 0x7FFFFFFFFFFFFF).astype(np.float32)
ch1_kurt = (cc1 >> 55).astype(np.float32)
del cc1

del chan0
del chan1

joinedChannels = np.hstack((np.fliplr(ch0_data), ch1_data)).T

lo = 0 * joinedChannels.shape[1] // 3
hi = 1 * joinedChannels.shape[1] // 3

jc = joinedChannels[:, lo:hi]

mn = jc.mean(axis=1)
for i in range(0, jc.shape[0]):
    (jc[i])[jc[i] == 0] = mn[i]


st = jc.std(axis=1)

x = np.linspace(1000, 3000, st.shape[0])
y = st / mn

fig = go.Figure(data=go.Scatter(y=st/mn, mode='lines'))
fig.update_yaxes(type="log")
fig.show()

