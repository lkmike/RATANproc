import sys
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

import plotly.colors
from PIL import ImageColor


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
    ('state', '>u4'),
    ('channel', '>u4'),
    ('data', '<u8', 0x80)
])

# block_array = np.fromfile("/home/michael/Documents/data/astro/122021/22122021/sun/az-12", dtype=dt)
block_array = np.fromfile("/home/michael/Documents/astro/032022/250322/3C84/az-10", dtype=dt)
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

# print(np.shape(ch0_data))
#
# b = '\n'.join('\t'.join('%e' %x for x in y) for y in ch0_data)
#
# with open('/home/michael/Documents/data/astro/122021/22122021/sun/txt/az-12_ch0.txt', 'w') as f:
#     original_stdout = sys.stdout
#     sys.stdout = f  # Change the standard output to the file we created.
#     print(b)
#     sys.stdout = original_stdout
#     f.close()
#
# b = '\n'.join('\t'.join('%e' %x for x in y) for y in ch1_data)
#
# with open('/home/michael/Documents/data/astro/122021/22122021/sun/txt/az-12_ch1.txt', 'w') as f:
#     original_stdout = sys.stdout
#     sys.stdout = f  # Change the standard output to the file we created.
#     print(b)
#     sys.stdout = original_stdout
#     f.close()
#
# exit()

kurt_threshold = 200


def suppress_by_kurtosis(data, kurtosis, th=kurt_threshold):
    avg_kurt = np.apply_along_axis(np.average, 0, kurtosis)
    for i in range(0, len(avg_kurt)):
        if i < th:
            data[:, i] = np.zeros(len(data))


# suppress_by_kurtosis(ch0_data, ch0_kurt)
# suppress_by_kurtosis(ch1_data, ch1_kurt)


def merge_bands(a, nbands=2):
    p = len(a) // nbands
    t = np.zeros(p)
    for i in range(0, p):
        t[i] = np.sum(a[i * nbands: (i + 1) * nbands - 1])
    return t


# totals0 = np.apply_along_axis(merge_bands, 1, ch0_data)
# totals1 = np.apply_along_axis(merge_bands, 1, ch1_data)


def sm(a, npoints=100):
    return np.convolve(a, np.ones(npoints) / npoints, mode='valid')


# smooths0 = np.apply_along_axis(sm, 0, totals0)
# smooths1 = np.apply_along_axis(sm, 0, totals1)


# plt.plot(x, smooths1)
# plt.yscale('log')
# plt.grid(True)
# plt.show()


za = dict(
    exponentformat='power',
    type='log',
    dtick=1,
)

# joinedChannels = np.hstack((np.fliplr(ch0_data), ch1_data)).T
joinedChannels = np.fliplr(ch0_data).T

# For 032022/1803/3c84/az-12
# badChannels = list(range(10, 90)) + [102, 116, 119, 293, 294, 298] + list(range(316, 321)) + [324] \
#               + list(range(371, 402)) + list(range(420, 449)) + [457] + list(range(461, 472)) + [475] \
#               + list(range(479, 497))
#
# joinedChannels[badChannels] = np.zeros(joinedChannels.shape[1])
joinedChannels = np.apply_along_axis(lambda p: merge_bands(p, 8), 0, joinedChannels)

# print(joinedChannels.shape)
# exit()

reductionFactor = 2

nbands, npoints = np.shape(joinedChannels)
joinedChannels = joinedChannels.reshape(nbands, npoints // reductionFactor, -1)
jc = np.mean(joinedChannels, axis=2)

# jc = jc[:, 190:930]
# jc = jc.clip(max=0.35e9)

# jc = jc[]
#
# for i in range(0, jc.shape[0]):
#     if jc[i].std()/jc[i].mean() > 1000000:
#         jc[i] = np.zeros(jc.shape[1])

# print(jc.shape)
# exit()

# u, s, vh = np.linalg.svd(jc)
# print(u.shape, s.shape, vh.shape)

# exit()

# 00 01 02
# 03 04 05
#
# 10 11 12
# 13 14 15
#
# 20 21 22
# 23 24 25

print(jc.shape)

sampleLength = 16384 * 1024 / 2000000000 * reductionFactor
x = np.arange(0, jc.shape[1]) * sampleLength
y = np.linspace(1000, 3000, jc.shape[0])

# fig = go.Figure(data=go.Heatmap(
#     x=x,
#     y=y,
#     z=np.log10(jc),
#     colorscale='Rainbow',
#     zmin=6,
#     zmax=np.log10(3e8)
# ))

# fig = go.Figure()
# for i in range(0,4):
#     fig.add_trace(go.Scatter(y=vh[:, i]))

colors = list(get_color("Rainbow", np.linspace(0, 1, jc.shape[0])))
colors.reverse()
fig = go.Figure()
for i in range(0, jc.shape[0]):
    fig.add_trace(go.Scatter(x=x, y=jc[i], line=dict(color=colors[i])))

# The source expected at ~90 s before the end of the record
# fig.update_xaxes(range=[x[-1]-100, x[-1]-80])


fig.show()
