#!/usr/bin/python
from __future__ import print_function
import six

# import modules
import os # interacting with file systems
import json

# number modules
import numpy as np
import scipy.stats as sps
import pandas as pd

# plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns


### Data conversion functions ######################################################################

def read_cells_from_json(path_in):
    with open(path_in, 'r') as fin:
        Cells = json.load(fin)
    Cells_new = {}
    for cell_id, cell in Cells.items():
        Cells_new[cell_id] = dotdict(cell)
    return Cells_new

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def cells2df(Cells_dict, columns = None):
    '''
    Take cell data (a dicionary of Cell objects) and return a dataframe.

    rescale : boolean
        If rescale is set to True, then the 6 major parameters are rescaled by their mean.
    '''

    # columns to include
    if not columns:
        columns = ['fov', 'peak', 'birth_label',
               'birth_time', 'division_time',
               'sb', 'sd', 'width', 'delta', 'tau', 'elong_rate', 'septum_position']


    # Make dataframe for plotting variables
    Cells_df = pd.DataFrame(Cells_dict).transpose() # must be transposed so data is in columns
    # Cells_df = Cells_df.sort(columns=['fov', 'peak', 'birth_time', 'birth_label']) # sort for convinience
    Cells_df = Cells_df.sort_values(by=['fov', 'peak', 'birth_time', 'birth_label'])
    Cells_df = Cells_df[columns].apply(pd.to_numeric)

    return Cells_df

### Filtering functions ############################################################################
def find_cells_of_birth_label(Cells, label_num=1):
    '''Return only cells whose starting region label is given.
    If no birth_label is given, returns the mother cells.
    label_num can also be a list to include cells of many birth labels
    '''

    fCells = {} # f is for filtered

    if type(label_num) is int:
        label_num = [label_num]

    for cell_id in Cells:
        if Cells[cell_id].birth_label in label_num:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def find_cells_of_fov(Cells, FOVs=[]):
    '''Return only cells from certain FOVs.

    Parameters
    ----------
    FOVs : int or list of ints
    '''

    fCells = {} # f is for filtered

    if type(FOVs) is int:
        FOVs = [FOVs]

    fCells = {cell_id : cell_tmp for cell_id, cell_tmp in six.iteritems(Cells) if cell_tmp.fov in FOVs}

    return fCells

def find_cells_of_fov_and_peak(Cells, fov_id, peak_id):
    '''Return only cells from a specific fov/peak
    Parameters
    ----------
    fov_id : int corresponding to FOV
    peak_id : int correstonging to peak
    '''

    fCells = {} # f is for filtered

    for cell_id in Cells:
        if Cells[cell_id].fov == fov_id and Cells[cell_id].peak == peak_id:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def find_cells_born_before(Cells, born_before=None):
    '''
    Returns Cells dictionary of cells with a birth_time before the value specified
    '''

    if born_before == None:
        return Cells

    fCells = {cell_id : Cell for cell_id, Cell in six.iteritems(Cells) if Cell.birth_time <= born_before}

    return fCells

def find_cells_born_after(Cells, born_after=None):
    '''
    Returns Cells dictionary of cells with a birth_time after the value specified
    '''

    if born_after == None:
        return Cells

    fCells = {cell_id : Cell for cell_id, Cell in six.iteritems(Cells) if Cell.birth_time >= born_after}

    return fCells

def organize_cells_by_channel(Cells, specs):
    '''
    Returns a nested dictionary where the keys are first
    the fov_id and then the peak_id (similar to specs),
    and the final value is a dictionary of cell objects that go in that
    specific channel, in the same format as normal {cell_id : Cell, ...}
    '''

    # make a nested dictionary that holds lists of cells for one fov/peak
    Cells_by_peak = {}
    for fov_id in specs.keys():
        Cells_by_peak[fov_id] = {}
        for peak_id, spec in specs[fov_id].items():
            # only make a space for channels that are analyized
            if spec == 1:
                Cells_by_peak[fov_id][peak_id] = {}

    # organize the cells
    for cell_id, Cell in Cells.items():
        try:
            Cells_by_peak[Cell.fov][Cell.peak][cell_id] = Cell
        except KeyError:
            pass

    # remove peaks and that do not contain cells
    remove_fovs = []
    for fov_id, peaks in six.iteritems(Cells_by_peak):
        remove_peaks = []
        for peak_id in peaks.keys():
            if not peaks[peak_id]:
                remove_peaks.append(peak_id)

        for peak_id in remove_peaks:
            peaks.pop(peak_id)

        if not Cells_by_peak[fov_id]:
            remove_fovs.append(fov_id)

    for fov_id in remove_fovs:
        Cells_by_peak.pop(fov_id)

    return Cells_by_peak

def filter_by_stat(Cells, center_stat='mean', std_distance=3):
    '''
    Filters a dictionary of Cells by ensuring all of the 6 major parameters are
    within some number of standard deviations away from either the mean or median
    '''

    # Calculate stats.
    Cells_df = cells2df(Cells)
    stats_columns = ['sb', 'sd', 'delta', 'elong_rate', 'tau', 'septum_position']
    cell_stats = Cells_df[stats_columns].describe()

    # set low and high bounds for each stat attribute
    bounds = {}
    for label in stats_columns:
        low_bound = cell_stats[label][center_stat] - std_distance*cell_stats[label]['std']
        high_bound = cell_stats[label][center_stat] + std_distance*cell_stats[label]['std']
        bounds[label] = {'low' : low_bound,
                         'high' : high_bound}

    # add filtered cells to dict
    fCells = {} # dict to hold filtered cells

    for cell_id, Cell in six.iteritems(Cells):
        benchmark = 0 # this needs to equal 6, so it passes all tests

        for label in stats_columns:
            attribute = getattr(Cell, label) # current value of this attribute for cell
            if attribute > bounds[label]['low'] and attribute < bounds[label]['high']:
                benchmark += 1

        if benchmark == 6:
            fCells[cell_id] = Cells[cell_id]

    return fCells

def find_last_daughter(cell, Cells):
    '''Finds the last daughter in a lineage starting with a earlier cell.
    Helper function for find_continuous_lineages'''

    # go into the daugther cell if the daughter exists
    if cell.daughters[0] in Cells:
        cell = Cells[cell.daughters[0]]
        cell = find_last_daughter(cell, Cells)
    else:
        # otherwise just give back this cell
        return cell

    # finally, return the deepest cell
    return cell

def lineages_to_dict(Lineages):
    '''Converts the lineage structure of cells organized by peak back
    to a dictionary of cells. Useful for filtering but then using the
    dictionary based plotting functions'''

    Cells = {}

    for fov, peaks in six.iteritems(Lineages):
        for peak, cells in six.iteritems(peaks):
            Cells.update(cells)

    return Cells

def find_continuous_lineages(Cells, specs, t1=0, t2=1000):
    '''
    Uses a recursive function to only return cells that have continuous
    lineages between two time points. Takes a "lineage" form of Cells and
    returns a dictionary of the same format. Good for plotting
    with saw_tooth_plot()

    t1 : int
        First cell in lineage must be born before this time point
    t2 : int
        Last cell in lineage must be born after this time point
    '''

    Lineages = organize_cells_by_channel(Cells, specs)

    # This is a mirror of the lineages dictionary, just for the continuous cells
    Continuous_Lineages = {}

    for fov, peaks in six.iteritems(Lineages):
       # print("fov = {:d}".format(fov))
        # Create a dictionary to hold this FOV
        Continuous_Lineages[fov] = {}

        for peak, Cells in six.iteritems(peaks):
           # print("{:<4s}peak = {:d}".format("",peak))
            # sort the cells by time in a list for this peak
            cells_sorted = [(cell_id, cell) for cell_id, cell in six.iteritems(Cells)]
            cells_sorted = sorted(cells_sorted, key=lambda x: x[1].birth_time)

            # Sometimes there are not any cells for the channel even if it was to be analyzed
            if not cells_sorted:
                continue

            # look through list to find the cell born immediately before t1
            # and divides after t1, but not after t2
            for i, cell_data in enumerate(cells_sorted):
                cell_id, cell = cell_data
                if cell.birth_time < t1 and t1 <= cell.division_time < t2:
                    first_cell_index = i
                    break

            # filter cell_sorted or skip if you got to the end of the list
            if i == len(cells_sorted) - 1:
                continue
            else:
                cells_sorted = cells_sorted[i:]

            # get the first cell and its last contiguous daughter
            first_cell = cells_sorted[0][1]
            last_daughter = find_last_daughter(first_cell, Cells)

            # check to the daughter makes the second cut off
            if last_daughter.birth_time > t2:
                # print(fov, peak, 'Made it')

                # now retrieve only those cells within the two times
                # use the function to easily return in dictionary format
                Cells_cont = find_cells_born_after(Cells, born_after=t1)
                # Cells_cont = find_cells_born_before(Cells_cont, born_before=t2)

                # append the first cell which was filtered out in the above step
                Cells_cont[first_cell.id] = first_cell

                # and add it to the big dictionary
                Continuous_Lineages[fov][peak] = Cells_cont

        # remove keys that do not have any lineages
        if not Continuous_Lineages[fov]:
            Continuous_Lineages.pop(fov)

    Cells = lineages_to_dict(Continuous_Lineages) # revert back to return

    return Cells

def binned_stat(x, y, statistic='mean', bin_edges='sturges', binmin=None):
    '''Calculate binned mean or median on X. Returns plotting variables

    bin_edges : int or list/array
        If int, this is the number of bins. If it is a list it defines the bin edges.

    '''

    # define range for bins
    data_mean = x.mean()
    data_std = x.std()
    bin_range = (data_mean - 3*data_std, data_mean + 3*data_std)

    # gives better bin edges. If a defined sequence is passed it will use that.
    bin_edges = np.histogram_bin_edges(x, bins=bin_edges, range=bin_range)

    # calculate mean
    bin_result = sps.binned_statistic(x, y,
                                      statistic=statistic, bins=bin_edges)
    bin_means, bin_edges, bin_n = bin_result
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    # calculate error at each bin (standard error)
    bin_error_result = sps.binned_statistic(x, y,
                                            statistic=np.std, bins=bin_edges)
    bin_stds, _, _ = bin_error_result

    # if using median, multiply this number by 1.253. Holds for large samples only
    if statistic == 'median':
        bin_stds = bin_stds * 1.253

    bin_count_results = sps.binned_statistic(x, y,
                                             statistic='count', bins=bin_edges)
    bin_counts, _, _ = bin_count_results

    bin_errors = np.divide(bin_stds, np.sqrt(bin_counts))

    # remove bins with not enought datapoints
    if binmin:
        delete_me = []
        for i, points in enumerate(bin_counts):
            if points < binmin:
                delete_me.append(i)
        delete_me = tuple(delete_me)
        bin_centers = np.delete(bin_centers, delete_me)
        bin_means = np.delete(bin_means, delete_me)
        bin_errors = np.delete(bin_errors, delete_me)

        # only keep locations where there is data
        bin_centers = bin_centers[~np.isnan(bin_means)]
        bin_means = bin_means[~np.isnan(bin_means)]
        bin_errors = bin_errors[~np.isnan(bin_means)]

    return bin_centers, bin_means, bin_errors


### Plotting functions #############################################################################

def plot_channel_traces(Cells, time_int=1.0, fl_plane='c2', alt_time='birth',
                        fl_int=1.0, plot_fl=False, plot_foci=False, plot_pole=False,
                        pxl2um=1.0, xlims=None, foci_size=100):
    '''Plot a cell lineage with profile information. Plots cells at their Y location in the growth channel.

    Parameters
    ----------
    Cells : dict of Cell objects
        All the cells should come from a single peak.
    time_int : int or float
        Used to adjust the X axis to plot in hours
    alt_time : float or 'birth'
        Adjusts all time by this value. 'birth' adjust the time so first birth time is at zero.
    fl_plane : str
        Plane from which to get florescent data
    plot_fl : boolean
        Flag to plot florescent line profile.
    plot_foci : boolean
        Flag to plot foci or not.
    plot_pole : boolean
        If true, plot different colors for cells with different pole ages.
    plx2um : float
        Conversion factor between pixels and microns.
    xlims : [float, float]
        Manually set xlims. If None then set automatically.
    '''

    time_int = float(time_int)
    fl_int = float(fl_int)

    y_adj_px = 3 # number of pixels to adjust down y profile
    color = 'b' # overwritten if plot_pole == True

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 3))
    ax = [axes]

    # turn it into a list to fidn first time
    lin = [(cell_id, cell) for cell_id, cell in six.iteritems(Cells)]
    lin = sorted(lin, key=lambda x: x[1].birth_time)

    # align time to first birth or shift time
    if alt_time == None:
        alt_time = 0
    elif alt_time == 'birth':
        alt_time = lin[0][1].birth_time * time_int / 60.0

    # determine last time for xlims
    if xlims == None or xlims[1] == None:
        if alt_time == 'birth' or alt_time == 0:
            first_time = 0
        else: # adjust for negative birth times
            first_time = (lin[0][1].times[0] - 10) * time_int / 60.0 - alt_time
        last_time = (lin[-1][1].times[-1] + 10) * time_int / 60.0 - alt_time
        xlims = (first_time, last_time)

    # adjust scatter marker size so colors touch but do not overlap
    # uses size of figure in inches, with the dpi (ppi) to convert to points.
    # scatter marker size is points squared.
    bbox = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = float(bbox.width), float(bbox.height)
    # print(fig.dpi, width, xlims[1], xlims[0],  time_int)
    # print(((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int)))
    # print((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0))
    # print((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int)**2)
    scat_s = (((fig.dpi * width) / ((xlims[1] - xlims[0]) * 60.0 / time_int) * fl_int))**2
    # print(time_int)
    # print(scat_s)

    # Choose colormap. Need to add alpha to color map and normalization
    # green/c2
    if plot_fl:
        max_c2_int = 0
        min_c2_int = float('inf')
        for cell_id, cell in lin:
            for profile_t in getattr(cell, 'fl_profiles_' + fl_plane):
                if max(profile_t) > max_c2_int:
                    max_c2_int = max(profile_t)
                if min(profile_t) < min_c2_int:
                    min_c2_int = min(profile_t)
        cmap_c2 = plt.cm.Greens
        color_norm_c2 = mpl.colors.Normalize(vmin=min_c2_int, vmax=max_c2_int)

    for cell_id, cell in six.iteritems(Cells):

        # if this is a complete cell plot till division with a line at the end
        cell_times = np.array(cell.times) * time_int / 60.0 - alt_time
        cell_yposs = np.array([y for y, x in cell.centroids]) * pxl2um
        cell_halflengths = np.array(cell.lengths) / 2.0 * pxl2um
        ytop = cell_yposs + cell_halflengths
        ybot = cell_yposs - cell_halflengths

        if plot_pole:
            if cell.poleage:
                color_choices = sns.hls_palette(4)
                if cell.poleage == (1000, 0):
                    color = color_choices[0]
                elif cell.poleage == (0, 1) and cell.birth_label <= 2:
                    color = color_choices[1]
                elif cell.poleage == (1, 0) and cell.birth_label <= 3:
                    color = color_choices[2]
                elif cell.poleage == (0, 2):
                    color = color_choices[3]
                # elif cell.poleage == (2, 0):
                #     color = color_choices[4]
                else:
                    color = 'k'
            elif cell.poleage == None:
                    color = 'k'

        # plot two lines for top and bottom of cell
        ax[0].plot(cell_times, ybot, cell_times, ytop,
                   color=color, alpha=0.75, lw=1)
        # ax[0].fill_between(cell_times, ybot, ytop,
        #                    color=color, lw=0.5, alpha=1)

        # plot lines for birth and division
        ax[0].plot([cell_times[0], cell_times[0]], [ybot[0], ytop[0]],
                      color=color, alpha=0.75, lw=1)
        ax[0].plot([cell_times[-1], cell_times[-1]], [ybot[-1], ytop[-1]],
                      color=color, alpha=0.75, lw=1)

        # plot fluorescence line profile
        if plot_fl:
            for i, t in enumerate(cell_times):
                if cell.times[i] % fl_int == 1:
                    fl_x = np.ones(len(getattr(cell, 'fl_profiles_' + fl_plane)[i])) * t # times
                    fl_ymin = cell_yposs[i] - (len(getattr(cell, 'fl_profiles_' + fl_plane)[i])/2 * pxl2um)
                    fl_ymax = fl_ymin + (len(getattr(cell, 'fl_profiles_' + fl_plane)[i]) * pxl2um)
                    fl_y = np.linspace(fl_ymin, fl_ymax, len(getattr(cell, 'fl_profiles_' + fl_plane)[i]))
                    fl_z = getattr(cell, 'fl_profiles_' + fl_plane)[i]
                    ax[0].scatter(fl_x, fl_y, c=fl_z, cmap=cmap_c2,
                                  marker='s', s=scat_s, norm=color_norm_c2,
                                  rasterized=True)

        # plot foci
        if plot_foci:
            for i, t in enumerate(cell_times):
                if cell.times[i] % fl_int == 1:
                    for j, foci_y in enumerate(cell.disp_l[i]):
                        foci_y_pos = cell_yposs[i] + (foci_y * pxl2um)
                        ax[0].scatter(t, foci_y_pos,
                                       s=cell.foci_h[i][j]/foci_size, linewidth=0.5,
                                       edgecolors='k', facecolors='none', alpha=0.5,
                                       rasterized=False)

    ax[0].set_xlabel('time (hours)')
    ax[0].set_xlim(xlims)
#     ax[0].set_ylabel('position ' + pnames['um'])
    ax[0].set_ylim([0, None])
#     ax[0].set_yticklabels([0,2,4,6,8,10])
    sns.despine()
    plt.tight_layout()

def plot_moving_avg(df,time_mark,column,window,ax,label=None):
    time_df = df[[time_mark, column]].apply(pd.to_numeric)
    xlims = (time_df[time_mark].min(), time_df[time_mark].max()) # x lims for bins
    # xlims = x_extents
    bin_mean, bin_edges, bin_n = sps.binned_statistic(time_df[time_mark], time_df[column],
                    statistic='mean', bins=np.arange(xlims[0]-1, xlims[1]+1, window))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    ax.plot(bin_centers, bin_mean, lw=1, alpha=1,label=label)

def make_line_hist(data,bins=None,density=True):
    if bins is None:
        bin_vals, bin_edges = np.histogram(data,density=density)
    else:
        bin_vals, bin_edges = np.histogram(data,density=density,bins=bins)
    bin_steps = np.diff(bin_edges)/2.0
    bin_centers = bin_edges[:-1] + bin_steps
    # add zeros to the next points outside this so plot line always goes down
    bin_centers = np.insert(bin_centers, 0, bin_centers[0] - bin_steps[0])
    bin_centers = np.append(bin_centers, bin_centers[-1] + bin_steps[-1])
    bin_vals = np.insert(bin_vals, 0, 0)
    bin_vals = np.append(bin_vals, 0)
    return(bin_centers,bin_vals)

def plot_distributions(df, columns, labels= None, titles = None):

    fig, axes = plt.subplots(1,6,figsize=(12,3))
    ax = np.ravel(axes)

    if not labels:
        labels = ['Birth length ($\mu$M)','Division length ($\mu$M)','$\Delta$ ($\mu$M)','Elongation rate (1/hr)',
            '$\\tau$ (minutes)','Septum position']

    titles = ['S$_{B}$','S$_{D}$','$\Delta$','$\lambda$','$\\tau$','L$_{1/2}$']

    for i,c in enumerate(columns):
        mu1 = df[c].mean()
        cv1 = df[c].std()/df[c].mean()
        
        ax[i].set_title(titles[i],fontsize=14)
        b1, v1 = make_line_hist(df[c],density=True)
        ax[i].plot(b1,v1, ls='-',color='C0',
                label='$\mu$ = {:2.2f}\nCV = {:2.2f}'.format(mu1,cv1)
                ,lw=1)
        
        ax[i].set_xlabel(labels[i],fontsize=12)
        ax[i].set_ylim(0,np.max(v1)*1.3)
        ax[i].set_yticks([])
    #     ax[i].legend(frameon=False,fontsize=6,loc=1)

    sns.despine(left=True)
    plt.tight_layout()

def plot_hex_time(Cells_df, time_mark='birth_time', x_extents=None, bin_extents=None):
    '''
    Plots cell parameters over time using a hex scatter plot and a moving average
    '''

    # lists for plotting and formatting
    columns = ['sb', 'elong_rate', 'sd', 'tau', 'delta', 'septum_position']
    titles = ['Length at Birth', 'Elongation Rate', 'Length at Division',
              'Generation Time', 'Delta', 'Septum Position']
    ylabels = ['$\mu$m', '$\lambda$', '$\mu$m', 'min', '$\mu$m','daughter/mother']

    # create figure, going to apply graphs to each axis sequentially
    fig, axes = plt.subplots(nrows=3, ncols=2,
                             figsize=[8,8], squeeze=False)

    ax = np.ravel(axes)
    # binning parameters, should be arguments
    binmin = 3 # minimum bin size to display
    bingrid = (20, 10) # how many bins to have in the x and y directions
    moving_window = 10 # window to calculate moving stat

    # bining parameters for each data type
    # bin_extent in within which bounds should bins go. (left, right, bottom, top)
    if x_extents == None:
        x_extents = (Cells_df['birth_time'].min(), Cells_df['birth_time'].max())

    if bin_extents == None:
        bin_extents = [(x_extents[0], x_extents[1], 0, 4),
                      (x_extents[0], x_extents[1], 0, 1.5),
                      (x_extents[0], x_extents[1], 0, 8),
                      (x_extents[0], x_extents[1], 0, 140),
                      (x_extents[0], x_extents[1], 0, 4),
                      (x_extents[0], x_extents[1], 0, 1),
                      (x_extents[0], x_extents[1], 0, 100),
                      (x_extents[0], x_extents[1], 0, 80),
                      (x_extents[0], x_extents[1], 0, 2)]

    # Now plot the filtered data
    for i, column in enumerate(columns):
        # get out just the data to be plot for one subplot
        time_df = Cells_df[[time_mark, column]].apply(pd.to_numeric)
        time_df.sort_values(by=time_mark, inplace=True)

        # plot the hex scatter plot
        p = ax[i].hexbin(time_df[time_mark], time_df[column],
                         mincnt=binmin, gridsize=bingrid)

        # graph moving average
        # xlims = (time_df['birth_time'].min(), time_df['birth_time'].max()) # x lims for bins
        xlims = x_extents
        try:
            bin_mean, bin_edges, bin_n = sps.binned_statistic(time_df[time_mark], time_df[column],
                        statistic='mean', bins=np.arange(xlims[0]-1, xlims[1]+1, moving_window))
            bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
            ax[i].plot(bin_centers, bin_mean, lw=4, alpha=0.8, color='yellow')

        except:
            pass

        # formatting
        ax[i].set_title(titles[i])
        ax[i].set_ylabel(ylabels[i])

        p.set_cmap(cmap=plt.cm.Blues) # set color and style

    ax[5].set_xlabel('%s [frame]' % time_mark)
    ax[4].set_xlabel('%s [frame]' % time_mark)

    plt.tight_layout()

    return fig, ax