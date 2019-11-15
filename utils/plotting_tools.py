import numpy as np

from matplotlib.collections import LineCollection
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class PlottingTools():
    def __init__(self):
        pass

    def multiple_barplots(self, data, x_labels, panel_row_count, panel_column_count, filename,
                          fig_height, fig_width, colors=None, ylimits=None, verbose=False, pdf=True, summary_data=False, set_title=[], error_bars=True, output_dir=None):

        fig, axs = plt.subplots(panel_row_count, panel_column_count)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        all_count = panel_row_count * panel_column_count
        for idx in range(all_count):
            if summary_data == True:
                #Outer conditions - inner conditions - mean sem
                means = data[idx, :, 0]
                high_ci = means + data[idx, :, 1]
                low_ci = means - data[idx, :, 1]
                show_error_bars = True and error_bars
            else:
                means = np.array([np.mean(x) for x in data[:, idx, :]])
                if data.shape[2] == 1: # if it is just mean values
                    show_error_bars = False
                else:
                    high_ci = np.array([np.percentile(x, 97.5) for x in data[:, idx, :]])
                    low_ci = np.array([np.percentile(x, 2.5) for x in data[:, idx, :]])
                    show_error_bars = True and error_bars

            if verbose == True:
                print(means)

            if all_count == 1:
                ax = axs
            else:
                ax = axs[idx]
            xs = np.arange(means.shape[0])

            if show_error_bars:
                if colors is None:
                    ax.bar(xs, means,
                           yerr=np.vstack([high_ci - means, means - low_ci]))
                else:
                    ax.bar(xs, means,
                           yerr=np.vstack([high_ci - means, means - low_ci]),
                           color=colors)
            else:
                ax.bar(xs, means,
                       color=colors)


            ax.tick_params(labelsize=15)
            if x_labels is not None:
                ax.set_xticks(xs)
                ax.set_xticklabels(x_labels, rotation=40)
            else:
                ax.set_xticks([])
            if ylimits is not None:
                ax.set_ylim(ylimits)
            else:
                ax.set_ylim((0, 1))
            if idx > 0:
                ax.set_yticks([])

            if set_title != []:
                ax.set_title(set_title[idx])


        all_axes = fig.get_axes()

        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if output_dir == None:
            output_dir = './output/'
        if pdf == True:
            plot_path = output_dir + filename + '.pdf'
        else:
            plot_path = output_dir + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    def multiple_lines(self, means, sems, labels, filename, fig_height, fig_width, y_limits=None, verbose=False, colors=None, linetypes=None, markers=None, stepsize=None, start=None, end=None, shade=None, pdf=True, output_dir=None):
        means = np.array(means)
        sems = np.array(sems)
        fig, axs = plt.subplots(1, 1)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        if verbose == True:
            print(means)
        
        ax = axs
        xs = np.arange(means.shape[1])


        for s in range(means.shape[0]):
            if shade[s] == True:
                ax.plot(xs, means[s], color=colors[s], linestyle=linetypes[s], marker=markers[s])
                ax.fill_between(xs, means[s] - sems[s], means[s] + sems[s],
                                alpha=0.25, edgecolor='white', facecolor=colors[s])
            else:
                ax.errorbar(xs, means[s],
                            yerr=np.vstack([sems[s], sems[s]]),
                            color=colors[s], linestyle=linetypes[s],
                            marker=markers[s])
            

        """
        for s in range(means.shape[0]):
            if colors is not None:
                ax.errorbar(xs, means[s],
                            yerr=np.vstack([sems[s], sems[s]]),
                            color=colors[s], linestyle=linetypes[s],
                            marker=markers[s])
            else:
                ax.errorbar(xs, means[s],
                            yerr=np.vstack([sems[s], sems[s]]))
        """
        ax.tick_params(labelsize=12)
        ax.set_xticks(range(means.shape[1]))
        if labels is not None:
            ax.set_xticklabels(labels)

        if y_limits is not None:
            ax.set_ylim(y_limits)

        #start, end = ax.get_xlim()
        if stepsize is not None:
            ax.xaxis.set_ticks(np.arange(start, end, stepsize))

        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if output_dir == None:
            output_dir = './output/'
        if pdf == True:
            plot_path = output_dir + filename + '.pdf'
        else:
            plot_path = output_dir + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


    def multiple_rsa_matrices(self, data, panel_row_count, panel_column_count,
                              filename, fig_height, fig_width, normalize=True, pdf=True, output_dir=None):
        
        N =data[0].shape[0]
        off_diag = 1 - np.eye(N)

        fig, axs = plt.subplots(panel_row_count, panel_column_count)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        all_count = panel_row_count * panel_column_count
        for idx, matrix in enumerate(data):

            if normalize == True:
                matrix[off_diag == 1] = (matrix[off_diag == 1] - np.min(matrix[off_diag == 1])) / (np.max(matrix[off_diag == 1]) - np.min(matrix[off_diag == 1]))
                minn = 0
                maxx = 1
            else:
                minn = -1
                maxx = 1

            #matrix[np.eye(N) == 1] = np.max(matrix[off_diag == 1])

            ax = axs[idx]
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            #im = ax.imshow(matrix[::-1,:], cmap='Greys')
            im = ax.pcolormesh(matrix, vmin=0., vmax=1., cmap='Greys')


        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)

        if output_dir == None:
            output_dir = './output/'
        if pdf == True:
            plot_path = output_dir + filename + '.pdf'
        else:
            plot_path = output_dir + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)



    def scatter_plot(self, data_1, data_2, filename, fig_height, fig_width, mask=None, colors=None, verbose=False, pdf=True, set_limits=False, output_dir=None):
        fig, axs = plt.subplots(1, 1)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        if verbose == True:
            print(np.corrcoef(data_1.flatten(), data_2.flatten())[0,1])
        
        ax = axs
        ax.tick_params(labelsize=20)

        if mask is not None:
            elements = np.max(mask) + 1
            for elem in range(elements):
                #inverse ordering color
                ax.scatter(data_1[mask == elem], data_2[mask == elem], color=colors[elements - 1 - elem], s=200) #use s=200 for model vs. data coefficients plots
        else:
            ax.scatter(data_1, data_2)

        if set_limits == True:
            ax.set_xlim((-0.02, 0.375))
            ax.set_ylim((-0.02, 0.375))
        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if output_dir == None:
            output_dir = './output/'

        if pdf == True:
            plot_path = output_dir + filename + '.pdf'
        else:
            plot_path = output_dir + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


    def scatter_plot_multipane(self, data_1, data_2, filename, fig_height, fig_width, mask=None, colors=None, verbose=False, pdf=True, set_limits=False, submask=None, output_dir=None):

        panel_row_count = np.max(mask) + 1
        fig, axs = plt.subplots(panel_row_count, panel_row_count)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        panel_counter = 0
        for i in range(panel_row_count):
            for j in range(panel_row_count):
                ax = axs[i, j]
                
                if submask is not None:
                    a = data_1[mask == i]
                    b = data_2[mask == j]
                    for k in [3, 0, 1, 2]:
                        ax.scatter(a[submask == k], b[submask == k], color=colors[k], s=.05)
                else:
                    ax.scatter(data_1[mask == i], data_2[mask == j], color=colors[2], s=5)

        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if output_dir == None:
            output_dir = './output/'

        if pdf == True:
            plot_path = output_dir + filename + '.pdf'
        else:
            plot_path = output_dir + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
