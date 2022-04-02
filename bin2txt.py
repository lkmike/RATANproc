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
    ('state', '<u4'),
    ('channel', '<u4'),
    ('data', '<u8', 0x80)
])

base_dir = "/home/michael/Documents/data/astro/032022/1703/sun/"
file_names = ["az+28", "az+24", "az+20", "az+16", "az+12", "az+08", "az+04", "az+00", "az-04", "az-08", "az-12",
              "az-16", "az-20", "az-24", "az-28"]

for file_name in file_names:
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


    def write_array(a_name, f_name):
        b = '\n'.join('\t'.join('%e' % x for x in y) for y in globals()[a_name])
        with open(f_name + '_' + a_name + '.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f  # Change the standard output to the file we created.
            print(b)
            sys.stdout = original_stdout
            f.close()


    write_array('ch0_pol0', base_dir + file_name)
    write_array('ch0_pol1', base_dir + file_name)
    write_array('ch1_pol0', base_dir + file_name)
    write_array('ch1_pol1', base_dir + file_name)

    write_array('ch0_pol0_kurt', base_dir + file_name)
    write_array('ch0_pol1_kurt', base_dir + file_name)
    write_array('ch1_pol0_kurt', base_dir + file_name)
    write_array('ch1_pol1_kurt', base_dir + file_name)

