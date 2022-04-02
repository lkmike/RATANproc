import sys
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd

import plotly.colors
from PIL import ImageColor

from sklearn.decomposition import TruncatedSVD

# base_dir = '/home/michael/Documents/data/astro/032022/1703/sun/'
# file_name = 'az+16'
base_dir = '/home/michael/Documents/data/astro/032022/1703/sun/'
file_name = 'az+16'
timeReductionFactor = 32
n_components = 6
quietSunPoint = 768



def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


dt = np.dtype([
    ('cnt', '<u4'),
    ('avg_kurt', '<u4'),
    ('state', '<u4'),
    ('channel', '<u4'),
    ('data', '<u8', 0x80)
])

cal_ch0_pol0 = np.genfromtxt('/home/michael/Documents/data/astro/032022/common/align_left1.txt')
cal_ch1_pol0 = np.genfromtxt('/home/michael/Documents/data/astro/032022/common/align_left2.txt')
cal_ch0_pol1 = np.genfromtxt('/home/michael/Documents/data/astro/032022/common/align_right1.txt')
cal_ch1_pol1 = np.genfromtxt('/home/michael/Documents/data/astro/032022/common/align_right2.txt')

cal = np.genfromtxt('/home/michael/Documents/data/astro/032022/common/Flux_Dm.txt')


def avg(a, n_avg=2):
    return np.mean(a.reshape(-1, n_avg), axis=1)


cal_ch0_pol0 = avg(cal_ch0_pol0)
cal_ch1_pol0 = avg(cal_ch1_pol0)
cal_ch0_pol1 = avg(cal_ch0_pol1)
cal_ch1_pol1 = avg(cal_ch1_pol1)

# Filter bands
cal_ch0_pol0[41:117] = 0
cal_ch0_pol1[41:117] = 0
cal_ch1_pol0[46:112] = 0
cal_ch1_pol1[46:112] = 0
cal_ch1_pol0[276:361] = 0
cal_ch1_pol1[276:361] = 0


# fig = go.Figure()
# fig.add_trace(go.Scatter(y=cal_ch0_pol0))
# fig.add_trace(go.Scatter(y=cal_ch1_pol0))
# fig.add_trace(go.Scatter(y=cal_ch0_pol1))
# fig.add_trace(go.Scatter(y=cal_ch1_pol1))
# fig.show()
# exit()

block_array = np.fromfile(base_dir + file_name, dtype=dt)
count = len(block_array)

ch0_pol0_max = 0
ch0_pol1_max = 0
ch1_pol0_max = 0
ch1_pol1_max = 0
pol_mask = 0b00000000000010000000000000000000
for el in block_array:
    if el['channel'] == 0:
        if (el['state'] & pol_mask) == 0:
            if el['cnt'] > ch0_pol0_max:
                ch0_pol0_max = el['cnt']
        else:
            if el['cnt'] > ch0_pol1_max:
                ch0_pol1_max = el['cnt']
    else:
        if (el['state'] & pol_mask) == 0:
            if el['cnt'] > ch1_pol0_max:
                ch1_pol0_max = el['cnt']
        else:
            if el['cnt'] > ch1_pol1_max:
                ch1_pol1_max = el['cnt']

avg_num = 2 ** (block_array[0]['avg_kurt'] & 0b111111)
spectrum_length = 8192 // avg_num


def align_to_spectrum_length(l):
    return ((l + 1) // 0x80) * 0x80


ch0_pol0_length = align_to_spectrum_length(ch0_pol0_max)
ch0_pol1_length = align_to_spectrum_length(ch0_pol1_max)
ch1_pol0_length = align_to_spectrum_length(ch1_pol0_max)
ch1_pol1_length = align_to_spectrum_length(ch1_pol1_max)

chan0_pol0 = np.zeros(ch0_pol0_length, dtype=dt)
chan0_pol1 = np.zeros(ch0_pol1_length, dtype=dt)
chan1_pol0 = np.zeros(ch1_pol0_length, dtype=dt)
chan1_pol1 = np.zeros(ch1_pol1_length, dtype=dt)

for el in block_array:
    if el['channel'] == 0:
        if ((el['state'] & pol_mask) == 0) and (el['cnt'] < ch0_pol0_length):
            chan0_pol0[el['cnt']] = el
        elif ((el['state'] & pol_mask) != 0) and (el['cnt'] < ch0_pol1_length):
            chan0_pol1[el['cnt']] = el
    else:
        if ((el['state'] & pol_mask) == 0) and (el['cnt'] < ch1_pol0_length):
            chan1_pol0[el['cnt']] = el
        elif ((el['state'] & pol_mask) != 0) and (el['cnt'] < ch1_pol1_length):
            chan1_pol1[el['cnt']] = el

del block_array


def get_data_and_kurtosis(a):
    cc = a['data'].reshape(-1, spectrum_length)
    return (cc & 0x7FFFFFFFFFFFFF).astype(np.float32), (cc >> 55).astype(np.float32)


ch0_pol0, ch0_pol0_kurt = get_data_and_kurtosis(chan0_pol0)
ch0_pol1, ch0_pol1_kurt = get_data_and_kurtosis(chan0_pol1)
ch1_pol0, ch1_pol0_kurt = get_data_and_kurtosis(chan1_pol0)
ch1_pol1, ch1_pol1_kurt = get_data_and_kurtosis(chan1_pol1)

print(ch0_pol0.shape)
print(ch1_pol0.shape)

ch0_pol0 = ch0_pol0 * cal_ch0_pol0
ch1_pol0 = ch1_pol0 * cal_ch1_pol0
ch0_pol1 = ch0_pol1 * cal_ch0_pol1
ch1_pol1 = ch1_pol1 * cal_ch1_pol1

# print(ch0_pol0.shape)
# exit()

ch0_pol0[ch0_pol0_kurt < 256] = np.NaN
ch1_pol0[ch1_pol0_kurt < 256] = np.NaN
ch0_pol1[ch0_pol1_kurt < 256] = np.NaN
ch1_pol1[ch1_pol1_kurt < 256] = np.NaN


def merge_bands(a, nbands=2):
    p = len(a) // nbands
    t = np.zeros(p)
    b = a
    # b[b == np.NaN] = 0
    for j in range(0, p):
        t[j] = np.sum(b[j * nbands: (j + 1) * nbands - 1])
    return t


# def replace_nan(a):
#     ok = ~np.isnan(a)
#     xp = ok.ravel().nonzero()[0]
#     fp = a[~np.isnan(a)]
#     xa = np.isnan(a).ravel().nonzero()[0]
#     a[np.isnan(a)] = np.interp(xa, xp, fp)
#     return a

def replace_nan(a):
    return pd.Series(a).interpolate().tolist()


# Yet will show only left polarization if both are present
if len(ch0_pol0) != 0:
    joinedChannels = np.hstack((np.fliplr(ch0_pol0), ch1_pol0)).T
elif len(ch0_pol1) != 0:
    joinedChannels = np.hstack((np.fliplr(ch0_pol1), ch1_pol1)).T

nBands, nPoints = np.shape(joinedChannels)
joinedChannels = joinedChannels.reshape(nBands, nPoints // timeReductionFactor, -1)
jc = np.nanmean(joinedChannels, axis=2)

jc = np.apply_along_axis(replace_nan, 0, jc)

jc[np.isnan(jc)] = 0

jc = jc[:, 200:1500]
jc[490:520] = np.zeros(jc.shape[1])
jc[97:127] = np.zeros(jc.shape[1])
jc[788] = np.zeros(jc.shape[1])
jc[558] = np.zeros(jc.shape[1])
jc[51] = np.zeros(jc.shape[1])
jc[71] = np.zeros(jc.shape[1])
jc[143] = np.zeros(jc.shape[1])
jc[153] = np.zeros(jc.shape[1])
jc[450:471] = np.zeros(jc.shape[1])
jc[395:402] = np.zeros(jc.shape[1])


svd = TruncatedSVD(n_components=n_components)
loadings = svd.fit_transform(jc.T)
scores = svd.components_

print("Loadings:", loadings.shape)
print("Scores:", scores.shape)

comp0 = np.outer(scores[0], loadings[:, 0])

print("Component:", comp0.shape)
print("jc:", jc.shape)
print("cal:", cal.shape)

jc = np.zeros(comp0.shape)
for i in range(0, n_components):
    jc += np.outer(scores[i], loadings[:, i])


for i in range(0, comp0.shape[0]):
    jc[i] *= cal[i] / (comp0[i, quietSunPoint] + 0.001)

loadings = svd.fit_transform(jc.T)
scores = svd.components_
jc = np.zeros(comp0.shape)
for i in range(0, n_components):
    jc += np.outer(scores[i], loadings[:, i])

sampleLength = 16384 * 1024 / 2000000000 * timeReductionFactor
x = np.arange(0, jc.shape[1]) * sampleLength

colors = list(get_color("Viridis", 0.1 + np.linspace(0, 1, loadings.shape[1] * 4//3)))
# colors.reverse()
fig = go.Figure()
for i in range(0, loadings.shape[1]):
    fig.add_trace(go.Scatter(x=x, y=(1 if i==0 else 5) * np.flip(loadings[:, i]), line=dict(width=4, color=colors[i])))

from PIL import Image
bkImage = Image.open("/home/michael/Documents/data/astro/032022/1703/magneto/20220317_081500_1024_HMIBC_rot2443cw.png")

fig.add_layout_image(
        dict(
            source=bkImage,
            xref="x domain",
            yref="y domain",
            x=0.26,
            y=0.7,
            sizex=0.4635*0.92975,
            sizey=1,
            sizing="contain",
            opacity=0.5,
            layer="below")
)
fig.update_layout(yaxis_range=[-300, 600])
fig.show()
exit()
# colors = list(get_color("Rainbow", np.linspace(0, 1, scores.shape[0])))
# colors.reverse()
fig = go.Figure()
for i in range(0, scores.shape[0]):
    fig.add_trace(go.Scatter(y=scores[i], line=dict(width=4, color=colors[i])))
fig.show()
# exit()

for i in [1,]:
    jc = np.outer(scores[i], loadings[:, i])

print(jc.shape)
print(jc)

#
# # For 28.03.22 -22
# # jc = jc[:, jc.shape[1]//2:]
# jc = np.clip(jc, a_min=1e6, a_max=0.4e9)

sampleLength = 16384 * 1024 / 2000000000 * timeReductionFactor
x = np.arange(0, jc.shape[1]) * sampleLength
y = np.linspace(1000, 3000, jc.shape[0])

colors = list(get_color("Rainbow", np.linspace(0, 1, jc.shape[0])))
colors.reverse()
fig = go.Figure()
for i in range(0, jc.shape[0]):
    fig.add_trace(go.Scatter(y=100*np.flip(jc[i]), line=dict(color=colors[i])))
fig.add_layout_image(
        dict(
            source=bkImage,
            xref="x domain",
            yref="y domain",
            x=0.23,
            y=0.94,
            sizex=0.4635,
            sizey=1,
            sizing="contain",
            opacity=0.5,
            layer="below")
)
fig.update_layout(yaxis_range=[-300, 600])
fig.show()
exit()

# fig = go.Figure(data=go.Heatmap(
#     x=x,
#     y=y,
#     z=np.log10(jc),
#     colorscale='Rainbow',
#     zmin=6,
#     zmax=np.log10(3e8)
# ))
# fig.show()
# exit()

# fig = go.Figure(data=go.Surface(
#     x=x,
#     y=y,
#     # z=np.log10(jc),
#     z=jc,
#     # z=jc,
#     colorscale='Rainbow',
#     showscale=False
#     # zmin=6,
#     # zmax=np.log10(3e8)
# ))
# fig.show()
