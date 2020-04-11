


########    IMPORTS    ########



from bokeh.models import Label
from bokeh.util.compiler import TypeScript



########    CLASSES    ########



class LatexLabel(Label):
    """A subclass of the Bokeh built-in `Label` that supports rendering
    LaTex using the KaTex typesetting library.

    Only the render method of LabelView is overloaded to perform the
    text -> latex (via katex) conversion. Note: ``render_mode="canvas``
    isn't supported and certain DOM manipulation happens in the Label
    superclass implementation that requires explicitly setting
    `render_mode='css'`).
    """
    __javascript__ = ["https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.js"]
    __css__ = ["https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.css"]

    TS_CODE = """
        import * as p from "core/properties"
        import {Label, LabelView} from "models/annotations/label"
        declare const katex: any

        export class LatexLabelView extends LabelView {
          model: LatexLabel

          render(): void {
            //--- Start of copied section from ``Label.render`` implementation

            // Here because AngleSpec does units tranform and label doesn't support specs
            let angle: number
            switch (this.model.angle_units) {
              case "rad": {
                angle = -this.model.angle
                break
              }
              case "deg": {
                angle = (-this.model.angle * Math.PI) / 180.0
                break
              }
              default:
                throw new Error("unreachable code")
            }

            const panel = this.panel != null ? this.panel : this.plot_view.frame

            const xscale = this.plot_view.frame.xscales[this.model.x_range_name]
            const yscale = this.plot_view.frame.yscales[this.model.y_range_name]

            let sx = this.model.x_units == "data" ? xscale.compute(this.model.x) : panel.xview.compute(this.model.x)
            let sy = this.model.y_units == "data" ? yscale.compute(this.model.y) : panel.yview.compute(this.model.y)

            sx += this.model.x_offset
            sy -= this.model.y_offset

            //--- End of copied section from ``Label.render`` implementation
            // Must render as superpositioned div (not on canvas) so that KaTex
            // css can properly style the text
            this._css_text(this.plot_view.canvas_view.ctx, "", sx, sy, angle)

            // ``katex`` is loaded into the global window at runtime
            // katex.renderToString returns a html ``span`` element
            katex.render(this.model.text, this.el, {displayMode: true})
          }
        }

        export namespace LatexLabel {
          export type Attrs = p.AttrsOf<Props>

          export type Props = Label.Props
        }

        export interface LatexLabel extends LatexLabel.Attrs {}

        export class LatexLabel extends Label {
          properties: LatexLabel.Props

          constructor(attrs?: Partial<LatexLabel.Attrs>) {
            super(attrs)
          }

          static init_LatexLabel() {
            this.prototype.default_view = LatexLabelView
          }
        }
        """

    __implementation__ = TypeScript(TS_CODE)



def get_scale(vals, base=10):
    """
    get the scaling factor resulting in at most n characters before the comma.

    Input:
        vals: (ndarray, list)
            the data to be scaled.
        base: (int)
            the base of the scale.

    Output:
        scale: (int)
            the value (elevated to base) defining the scale.
    """

    # get dependancies
    import numpy as np

    # check the entered data
    assert vals.__class__.__name__ in ['list', 'ndarray'], "'vals' must be a list or ndarray object."
    assert base.__class__.__name__ == "int", "'base' must be an int object."

    # get the scales
    return int(np.floor(np.log(np.max(abs(vals))) / np.log(base))) if np.max(abs(vals)) != 0 else 0



def plot_linker(ax, x1, y1, x2, y2, ytop, text=None, linekwargs={}, textkwargs={}):
    """
    plot a linker between point 1 and 2 in ax data adding the required text in between.

    Input:
        ax: (matplotlib.pyplot.Axis)
            the axis where the linker has to be plotted in.

        x1: (float, int)
            the x coordinate of the first point of the linker.

        y1: (float, int)
            the y coordinate of the first point of the linker.

        x2: (float, int)
            the x coordinate of the second point of the linker.

        y2: (float, int)
            the y coordinate of the second point of the linker.

        ytop: (float, int)
            the y coordinate of the horizontal bar to be plotted between point 1 and 2.

        text: (str)
            the text to be plotted.

        linekwargs: (dict)
            the line options that are passed to the plot function.

        textkwargs: (dict)
            the text options that are passed to the text function.

    Output:
        None. The linker is plotted directly within ax.
    """

    # plot the lines
    ax.plot([x1, x1], [y1, ytop], **linekwargs)
    ax.plot([x2, x2], [y2, ytop], **linekwargs)
    ax.plot([x1, x2], [ytop, ytop], **linekwargs)

    # plot the text if required
    tt = {'ha': "center", 'va': "bottom"}
    tt.update(textkwargs)
    ax.text(0.5 * (x1 + x2), ytop, text, **tt)



def set_layout(ax, x_label="", y_label_left="", y_label_right="", z_label="",
               title="", x_digits="{:0.2f}", y_digits="{:0.2f}", z_digits="{:0.2f}",
               x_tick_lim_sep=0.25, y_tick_lim_sep=0.25, z_tick_lim_sep=0.25,
               x_ticks_loc=None, x_ticks_lab=None,
               x_lab_va="top", x_lab_ha="center",
               x_ticks_lab_va="top", x_ticks_lab_ha="center",
               x_ticks_rotation=0, x_label_rotation=0,
               y_ticks_loc=None, y_ticks_lab=None,
               y_label_right_va="center", y_label_right_ha="center",
               y_label_left_va="center", y_label_left_ha="center",
               y_ticks_lab_va="center", y_ticks_lab_ha="right",
               y_ticks_rotation=0, y_label_left_rotation=90,
               y_label_right_rotation=270,
               z_ticks_loc=None, z_ticks_lab=None,
               z_lab_va="center", z_lab_ha="center",
               z_ticks_lab_va="center", z_ticks_lab_ha="center",
               z_ticks_rotation=0, z_label_rotation=0,
               label_size=10, label_color="k",
               x_ticks_text_size=8,
               y_ticks_text_size=8,
               z_ticks_text_size=8,
               ticks_text_color="k", axis_width=1,
               latitude=10, longitude=45, x_smart_ticks=True, y_smart_ticks=True):
    """
    function to automatically set the layout of a matplotlib axis object

    Input
        ax: (matplotlib axis object)
            the axis to be manipulated.
        x_label: (str)
            the label of the x axis.
        y_label_left: (str)
            the label of the y axis (on the left).
        y_label_right: (str)
            the label of the y axis (on the right).
        z_label: (str)
            the label of the z axis (if exists).
        title: (str)
            the label above the plot.
        x_tick_lim_sep: (float)
            ticks to axis limits separation.
        y_tick_lim_sep: (float)
            ticks to axis limits separation.
        z_tick_lim_sep: (float)
            ticks to axis limits separation.
        x_digits: (str)
            the format to be used for the x_ticks_labels.
        y_digits: (str)
            the format to be used for the y_ticks_labels.
        z_digits: (str)
            the format to be used for the z_ticks_labels.
        x_label_rotation: (float)
            the rotation of the x label.
        x_ticks_rotation: (float)
            the rotation of the x ticks labels.
        x_ticks_loc: (list or ndarray)
            an array with the positions of the ticks on the x axis.
        x_ticks_lab: (list or ndarray)
            the labels to be used for the ticks on the x axis.
        x_lab_va: (float)
            set the vertical alignment of the x label.
        x_lab_ha: (float)
            set the horizontal alignment of the x label.
        x_ticks_lab_va: (float)
            set the vertical alignment of the tick labels.
        x_ticks_lab_ha: (float)
            set the horizontal alignment of the tick labels.
        y_label_left_rotation: (float)
            the rotation of the y label left.
        y_label_right_rotation: (float)
            the rotation of the y label right.
        y_ticks_rotation: (float)
            the rotation of the y ticks labels.
        y_ticks_loc: (list or ndarray)
            an array with the positions of the ticks on the y axis.
        y_ticks_lab: (list or ndarray)
            the labels to be used for the ticks on the y axis.
        y_label_right_va: (float)
            set the vertical alignment of the y label right.
        y_label_right_ha: (float)
            set the horizontal alignment of the y label right.
        y_label_left_va: (float)
            set the vertical alignment of the y label left.
        y_label_left_ha: (float)
            set the horizontal alignment of the y label left.
        y_ticks_lab_va: (float)
            set the vertical alignment of the tick labels.
        y_ticks_lab_ha: (float)
            set the horizontal alignment of the tick labels.
        z_label_rotation: (float)
            the rotation of the z label.
        z_ticks_rotation: (float)
            the rotation of the z ticks labels.
        z_ticks_loc: (list or ndarray)
            an array with the positions of the ticks on the z axis.
        z_ticks_lab: (list or ndarray)
            the labels to be used for the ticks on the z axis.
        z_lab_va: (float)
            set the vertical alignment of the z label.
        z_lab_ha: (float)
            set the horizontal alignment of the z label.
        z_ticks_lab_va: (float)
            set the vertical alignment of the tick labels.
        z_ticks_lab_ha (float)
            set the horizontal alignment of the tick labels.
        label_size: (int)
            the size of the labels.
        label_color: (list, tuple, ndarray or str)
            the color of the labels.
        x_ticks_text_size: (int)
            the size of the text used on the x ticks.
        y_ticks_text_size: (int)
            the size of the text used on the y ticks.
        z_ticks_text_size: (int)
            the size of the text used on the z ticks.
        ticks_text_color: (list, tuple, ndarray or str)
            the color of the text used on the ticks.
        latitude: (float)
            the latitude angle for 3D plot output.
        longitude: (float)
            the longitude angle for 3D plot output.
        x_smart_ticks: (bool)
            should the x_bar be plotted up to the last tick? Otherwise it will be plotted up to the x_limits
        y_smart_ticks: (bool)
            should the y_bar be plotted up to the last tick? Otherwise it will be plotted up to the y_limits

    Output:
        the same axis with the modified layout
    """

    # import the required packages
    import numpy as np

    # internal function to extract the necessary values from data
    def getTicks(bnd, t_loc, t_lab, diff_bound, digits):
        """
        internal function to get the axis limits, bounds, ticks location and labels
        """

        # get the ticks number
        n = 5 if t_loc is None else len(t_loc)

        # get the axis limits
        edges = np.mean(np.diff(np.linspace(bnd[0], bnd[1], n))) * diff_bound

        # get the ticks location
        if t_loc is None: t_loc = np.linspace(bnd[0], bnd[1], n)

        # get the axis limits
        lim = [t_loc[0] - edges, t_loc[-1] + edges]

        # get the ticks labels
        if t_lab is None: t_lab = [digits.format(i) for i in t_loc]

        # return the output
        return lim, t_loc, t_lab

    # set the title
    try:
        ax.set_title(title, fontsize=label_size, color=label_color)
    except Exception:
        ax.suptitle(title, fontsize=label_size, color=label_color)

    # set the axis spines
    try:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(axis_width)
        ax.spines["bottom"].set_linewidth(axis_width)
    except Exception:
        for axis in ax.axes:
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.spines["left"].set_linewidth(axis_width)
            axis.spines["bottom"].set_linewidth(axis_width)

    # get the x axis limits, bounds, ticks position and labels
    if x_ticks_loc is not None:
        x_l = [np.min(x_ticks_loc), np.max(x_ticks_loc)]
    else:
        try:
            x_l = ax.xaxis.get_data_interval()
        except Exception:
            try:
                x_l = ax.axes.xaxis.get_data_interval()
            except Exception:
                x_l = ax.axes[0].xaxis.get_data_interval()
        if np.any(np.isinf(x_l)):
            try:
                x_l = [i.get_loc() for i in ax.xaxis.get_major_ticks()]
            except Exception:
                try:
                    x_l = [i.get_loc() for i in ax.axes.xaxis.get_major_ticks()]
                except Exception:
                    x_l = [i.get_loc() for i in ax.axes[0].xaxis.get_major_ticks()]
            x_l = np.array([np.min(x_l), np.max(x_l)])
    x_out, x_tlo, x_tla = getTicks(x_l, x_ticks_loc, x_ticks_lab, x_tick_lim_sep, x_digits)

    # set the x axis limits, ticks and labels
    try:
        ax.set_xlim(x_out)
        ax.set_xticks(x_tlo)
        ax.set_xticklabels(x_tla, fontsize=x_ticks_text_size, color=ticks_text_color, rotation=x_ticks_rotation,
                           va=x_ticks_lab_va, ha=x_ticks_lab_ha)
        ax.set_xlabel(x_label, fontsize=label_size, color=label_color, rotation=x_label_rotation, va=x_lab_va,
                      ha=x_lab_ha)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(width=axis_width)
    except Exception:
        try:
            ax.axes.set_xlim(x_out)
            ax.axes.set_xticks(x_tlo)
            ax.axes.set_xticklabels(x_tla, fontsize=x_ticks_text_size, color=ticks_text_color,
                                    rotation=x_ticks_rotation, va=x_ticks_lab_va, ha=x_ticks_lab_ha)
            ax.axes.set_xlabel(x_label, fontsize=label_size, color=label_color, rotation=x_label_rotation,
                                  va=x_lab_va, ha=x_lab_ha)
            ax.axes.xaxis.set_ticks_position("bottom")
            ax.axes.xaxis.set_tick_params(width=axis_width)
        except Exception:
            ax.axes[0].set_xlim(x_out)
            ax.axes[0].set_xticks(x_tlo)
            ax.axes[0].set_xticklabels(x_tla, fontsize=x_ticks_text_size, color=ticks_text_color,
                                       rotation=x_ticks_rotation, va=x_ticks_lab_va, ha=x_ticks_lab_ha)
            ax.axes[0].set_xlabel(x_label, fontsize=label_size, color=label_color, rotation=x_label_rotation,
                                  va=x_lab_va, ha=x_lab_ha)
            ax.axes[0].xaxis.set_ticks_position("bottom")
            ax.axes[0].xaxis.set_tick_params(width=axis_width)
    if x_smart_ticks:
        try:
            ax.spines["bottom"].set_bounds(x_tlo[0], x_tlo[-1])
        except Exception:
            for axis in ax.axes:
                axis.spines["bottom"].set_bounds(x_tlo[0], x_tlo[-1])


    # get the y axis limits, bounds, ticks position and labels
    if y_ticks_loc is not None:
        y_l = [np.min(y_ticks_loc), np.max(y_ticks_loc)]
    else:
        try:
            y_l = ax.yaxis.get_data_interval()
        except Exception:
            try:
                y_l = ax.axes.yaxis.get_data_interval()
            except Exception:
                y_l = ax.axes[0].yaxis.get_data_interval()
        if np.any(np.isinf(y_l)):
            try:
                y_l = [i.get_loc() for i in ax.yaxis.get_major_ticks()]
            except Exception:
                try:
                    y_l = [i.get_loc() for i in ax.axes.yaxis.get_major_ticks()]
                except Exception:
                    y_l = [i.get_loc() for i in ax.axes[0].yaxis.get_major_ticks()]
            y_l = np.array([np.min(y_l), np.max(y_l)])
    y_out, y_tlo, y_tla = getTicks(y_l, y_ticks_loc, y_ticks_lab, y_tick_lim_sep, y_digits)

    # set the y axis limits, ticks and labels
    try:
        ax.set_ylim(y_out)
        ax.set_yticks(y_tlo)
        ax.set_yticklabels(y_tla, fontsize=y_ticks_text_size, color=ticks_text_color, rotation=y_ticks_rotation,
                           va=y_ticks_lab_va, ha=y_ticks_lab_ha)
        ax.set_ylabel(y_label_left, fontsize=label_size, color=label_color, rotation=y_label_left_rotation,
                      va=y_label_left_va, ha=y_label_left_ha)
    except Exception:
        try:
            ax.axes.set_ylim(y_out)
            ax.axes.set_yticks(y_tlo)
            ax.axes.set_yticklabels(y_tla, fontsize=y_ticks_text_size, color=ticks_text_color, rotation=y_ticks_rotation,
                                    va=y_ticks_lab_va, ha=y_ticks_lab_ha)
            ax.axes.set_ylabel(y_label_left, fontsize=label_size, color=label_color, va=y_label_left_va,
                               ha=y_label_left_ha, rotation=y_label_left_rotation)
        except Exception:
            ax.axes[1].set_ylim(y_out)
            ax.axes[1].set_yticks(y_tlo)
            ax.axes[1].set_yticklabels(y_tla, fontsize=y_ticks_text_size, color=ticks_text_color,
                                       rotation=y_ticks_rotation, va=y_ticks_lab_va, ha=y_ticks_lab_ha)
            ax.axes[1].set_ylabel(y_label_left, fontsize=label_size, color=label_color, va=y_label_left_va,
                                  ha=y_label_left_ha, rotation=y_label_left_rotation)
    if y_smart_ticks:
        try:
            ax.spines["left"].set_bounds(y_tlo[0], y_tlo[-1])
        except Exception:
            for axis in ax.axes:
                axis.spines["left"].set_bounds(y_tlo[0], y_tlo[-1])
    try:
        try:
            ax.yaxis.set_ticks_position("left")
        except Exception:
            try:
                ax.axes.yaxis.set_ticks_position("left")
            except Exception:
                ax.axes[0].yaxis.set_ticks_position("left")
        z_l = None
    except Exception:

        # it"s a 3d plot
        # ax.yaxis.set_ticks_position("bottom")
        ax.set_ylabel(y_label_left, fontsize=label_size, color=label_color, rotation=y_label_left_rotation,
                      va=y_label_left_va, ha=y_label_left_ha)
        ax.set_yticklabels(y_tla, fontsize=y_ticks_text_size, va=y_ticks_lab_va, color=ticks_text_color,
                           ha=y_ticks_lab_ha, rotation=y_ticks_rotation)

        # ax.xaxis.set_ticks_position("bottom")
        ax.set_xticklabels(x_tla, fontsize=x_ticks_text_size, va=x_ticks_lab_va, color=ticks_text_color,
                           ha=x_ticks_lab_ha, rotation=x_ticks_rotation)
        ax.set_xlabel(x_label, fontsize=label_size, color=label_color, rotation=x_label_rotation, va=x_lab_va,
                      ha=x_lab_ha)
        if z_ticks_loc != "auto":
            z_l = [np.min(z_ticks_loc), np.max(z_ticks_loc)]
        else:
            z_l = ax.zaxis.get_data_interval()
    try:
        ax.yaxis.set_tick_params(width=axis_width)
    except Exception:
        try:
            ax.axes.yaxis.set_tick_params(width=axis_width)
        except Exception:
            ax.axes[0].yaxis.set_tick_params(width=axis_width)

    # get the z axis limits, bounds, ticks position and labels
    if z_l is not None:
        z_out, z_tlo, z_tla = getTicks(z_l, z_ticks_loc, z_ticks_lab, z_tick_lim_sep, z_digits)

        # set the z axis limits, ticks and labels
        ax.set_zlim(z_out)
        ax.set_zticks(z_tlo)
        ax.zaxis.set_tick_params(width=axis_width)
        ax.set_zticklabels(z_tla, fontsize=z_ticks_text_size, color=ticks_text_color, rotation=z_ticks_rotation,
                           va=z_ticks_lab_va, ha=z_ticks_lab_ha)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(z_label, fontsize=label_size, color=label_color, rotation=z_label_rotation, va=z_lab_va,
                      ha=z_lab_ha)
        ax.zaxis.set_tick_params(width=axis_width)

        # set the rotation of the plot (only for 3D plots)
        ax.view_init(latitude, longitude)
    else:
        ax.text(x_l[1], np.mean(y_l), y_label_right, fontsize=label_size, rotation=y_label_right_rotation,
                color=label_color, ha=y_label_right_ha, va=y_label_right_va)



def cm2in(X):
    """
    convert a value expressed in centimeters to inches

    Input
        X: (float)
            the number to be converted

    Output
        Y: (float)
            the same number converted in centimeters
    """

    # import the required packages
    from numpy import float

    return float(X) / 2.54



def get_colors_from_map(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, .. n-1 to a distinct RGB color. The keyword argument name must be a
    standard mpl colormap name.

    Input:
        n: (int)
            the number of colors from the map.
        name: (str)
            the name of the colormap.

    Output:
        colors: (colormap)
            a list of colors.
    '''

    # check input variables
    assert n.__class__.__name__[:3] == "int", "'n' must be an int."
    assert name.__class__.__name__ == "str", "'name' must be a string."

    # import dependancies
    import matplotlib.pyplot as pl
    from numpy import linspace

    # get the colormap
    cmap = pl.cm.get_cmap(name)

    # create the segmentation required
    x = linspace(0, 1, n)

    # return the list of colors
    return [cmap(i) for i in x]



def axis3D():
    '''
    generate a 3D axis where to plot data
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pl
    return pl.axes(projection='3d')



def getErrorbar(data, x_var, y_var, row_var, col_var, colors, err=None,
                x_ticks=None, x_ticks_n=5, y_ticks=None, y_ticks_n=5,
                x_label='', y_label='', plot_row_label=True,
                plot_col_label=True, label_dim=12, col_title_angle=0,
                number_dim=10, label_sep='\n', x_ticks_decimals=2,
                y_ticks_decimals=2, h_window=9, w_window=9, interp_line=True,
                shareX=True, shareY=True, shareAll=True, col_title_ha='center',
                col_title_va='center', row_title_angle=0, row_title_ha='left',
                row_title_va='center'):
    '''
    Function for the creation of a errorbar grid. It plot out the error bars as
    mean points for each x value plus 95% confidence intervals error bars.

    Input:
                  data = a pandas dataframe containing all data
                 x_var = the name of the data's column reflecting the x-axis
                         coordinates
                 y_var = a list containing the name of the data's columns
                         reflecting the y-axis coordinates
               row_var = a list containing the name of the data's columns
                         defining the rows of the grid
               col_var = a list containing the name of the data's columns
                         defining the columns of the grid
                colors = a list containing the names of the colors to be
                         associated to each y_var element. colors and y_var
                         must have the same length
                   err = the function to be used to calculate error bars
                         (numpy standard deviation by default)
               x_ticks = the ticks of the X axis (if None, they are calculated)
             x_ticks_n = the number of ticks of the X axis
               y_ticks = the ticks of the Y axis (if None, they are calculated)
             y_ticks_n = the number of ticks of the Y axis
               x_label = the label of the X axis
               y_label = the label of the Y axis
        plot_row_label = if True, the labels defining the rows are plotted.
        plot_col_label = if True, the labels defining the columns are plotted.
             label_dim = the size of the labels and text
            number_dim = the size of the axes numbers
             label_sep = the separator between multiple rows/columns variables
      x_ticks_decimals = the number of decimals for the X-axis ticks
      y_ticks_decimals = the number of decimals for the Y-axis ticks
              h_window = the size (in cm) of the output figure for each row
              w_window = the size (in cm) of the output figure for each column
           interp_line = if True add a cubic spline interpolated line
                shareX = share the X axis limits and bounds between columns
                shareY = share the Y axis limits and bounds between rows
              shareAll = share both the X and Y axes limits and bounds between
                         all the subplots. If True, it overcomes shareX and
                         shareY.
       col_title_angle = the angle to print the columns title,
          col_title_ha = the horizontal alignment of the columns title
          col_title_va = the vertical alignment of the columns title
       row_title_angle = the angle to print the rows title,
          row_title_ha = the horizontal alignment of the rows title
          row_title_va = the vertical alignment of the rows title
    Output:
        a matplotlib Figure object
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as pl
    import matplotlib.gridspec as gs
    import myModule.utils as myut
    from myModule.stats import ci
    from myModule.signalprocessing import interpolate

    # internal function to get the ybounds of a subplot
    def getYbounds(row_df, col_df, x_var, y_var, data):
        ybounds = []
        sub = pd.concat([row_df, col_df], 1).to_dict('list')
        x_val = np.unique(myut.extract(data, sub)[0][x_var].values)
        for x_v in x_val:
            sub[x_var] = [x_v]
            y_v = myut.extract(data, sub)[0]
            ybounds.append([ci(y_v[y].values) for y in y_var])
        ybounds = np.array(ybounds).flatten()
        return [np.min(ybounds), np.max(ybounds)]

    # internal function to get the xbounds of a subplot
    def getXbounds(row_df, col_df, x_var, data):
        sub = pd.concat([row_df, col_df], 1).to_dict('list')
        x_val = np.unique(myut.extract(data, sub)[0][x_var].values)
        try:
            return np.array([np.min(x_val), np.max(x_val)])
        except Exception:
            return np.arange(len(x_val))

    # get the general variables and the figure
    x_var = np.array(x_var).flatten()[0]
    row_val = {t: np.unique(data[t].values) for t in row_var}
    row_val = myut.getCombinations(row_val)
    col_val = {t: np.unique(data[t].values) for t in col_var}
    col_val = myut.getCombinations(col_val)
    grid = gs.GridSpec(row_val.shape[0], col_val.shape[0])
    y_num = '%.0' + str(y_ticks_decimals) + 'f'
    x_num = '%.0' + str(x_ticks_decimals) + 'f'
    figsize = (h_window * row_val.shape[0], w_window * col_val.shape[0])
    fig = pl.figure(figsize=[myut.cm2inches(t) for t in figsize])

    # get the bounds for the x and y axis if shareAll is True
    if shareAll:
        xbounds = []
        ybounds = []
        for row_i in np.arange(row_val.shape[0]):
            for col_i in np.arange(col_val.shape[0]):
                row_df = pd.DataFrame(row_val.iloc[row_i]).T
                row_df.index = [0]
                col_df = pd.DataFrame(col_val.iloc[col_i]).T
                col_df.index = [0]
                xbounds.append(getXbounds(row_df, col_df, x_var, data))
                ybounds.append(getYbounds(row_df, col_df, x_var, y_var, data))
        if x_ticks is None:
            xbounds = [np.min(xbounds), np.max(xbounds)]
        else:
            xbounds = [np.min(x_ticks), np.max(x_ticks)]
        if y_ticks is None:
            ybounds = [np.min(ybounds), np.max(ybounds)]
        else:
            ybounds = [np.min(y_ticks), np.max(y_ticks)]

    # plot the y calculated at each x in a rows X cols grid
    for row_i in np.arange(row_val.shape[0]):

        # if shareY is True and shareAll is False get the Y bounds
        if not shareAll and shareY:
            ybounds = []
            for col_i in np.arange(col_val.shape[0]):
                row_df = pd.DataFrame(row_val.iloc[row_i]).T
                row_df.index = [0]
                col_df = pd.DataFrame(col_val.iloc[col_i]).T
                col_df.index = [0]
                ybounds.append(getYbounds(row_df, col_df, x_var, y_var, data))
            if y_ticks is None:
                ybounds = [np.min(ybounds), np.max(ybounds)]
            else:
                ybounds = [np.min(y_ticks), np.max(y_ticks)]

        # setup each cell of the grid
        for col_i in np.arange(col_val.shape[0]):

            # get sub-data
            row_df = pd.DataFrame(row_val.iloc[row_i]).T
            row_df.index = [0]
            col_df = pd.DataFrame(col_val.iloc[col_i]).T
            col_df.index = [0]
            x_val = pd.concat([row_df, col_df], 1).to_dict('list')
            x_val = np.unique(myut.extract(data, x_val)[0][x_var].values)

            # if both shareY and shareAll are false get the Y bounds
            if not shareAll and not shareY:
                if y_ticks is None:
                    ybounds = getYbounds(row_df, col_df, x_var, y_var, data)
                else:
                    ybounds = [np.min(y_ticks), np.max(y_ticks)]

            # if shareX is True and shareAll is False, get the X bounds
            if not shareAll and shareX:
                xbounds = []
                for row_t in np.arange(row_val.shape[0]):
                    row_c = pd.DataFrame(row_val.iloc[row_t]).T
                    row_c.index = [0]
                    xbounds.append(getXbounds(row_c, col_df, x_var, data))
                if x_ticks is None:
                    xbounds = [np.min(xbounds), np.max(xbounds)]
                else:
                    xbounds = [np.min(x_ticks), np.max(x_ticks)]

            # if both shareX and shareAll are False, get the X bounds
            elif not shareAll and not shareX:
                if y_ticks is None:
                    ybounds = getYbounds(row_df, col_df, x_var, y_var, data)
                else:
                    ybounds = [np.min(y_ticks), np.max(y_ticks)]

            # get the yticks, yticklabels, ylimits and ylabel
            if y_ticks is None:
                yticks = np.linspace(ybounds[0], ybounds[1], y_ticks_n)
            else:
                yticks = y_ticks
            yticks = np.round(yticks, y_ticks_decimals)
            ydelta = abs(np.mean(np.diff(yticks))) * 0.25
            ylimits = (ybounds[0] - ydelta, ybounds[1] + ydelta)
            if shareAll or shareY and col_i > 0:
                yticklabels = ['' for t in yticks]
                ylabel = ''
            else:
                yticklabels = [y_num % t for t in yticks]
                ylabel = y_label

            # get the xticks, xticklabels and xlimits
            if x_ticks is None:
                xticks = np.linspace(xbounds[0], xbounds[1], x_ticks_n)
            else:
                xticks = x_ticks
            xticks = np.round(xticks, x_ticks_decimals)
            xdelta = abs(np.mean(np.diff(xticks))) * 0.25
            xlimits = (xbounds[0] - xdelta, xbounds[1] + xdelta)
            if shareAll or shareX and row_i < row_val.shape[0] - 1:
                xticklabels = ['' for t in xticks]
                xlabel = ''
            else:

                # handle non numeric x labels
                if x_val.dtype.type is np.string_:
                    xticklabels = pd.concat([row_df, col_df], 1)
                    xticklabels = myut.extract(data, xticklabels)[0]
                    xticklabels = np.unique(xticklabels[x_var].values)
                else:
                    xticklabels = [x_num % t for t in xticks]
                xlabel = x_label

            # get the axis subplot
            ax = fig.add_subplot(grid[row_i, col_i])

            # set the x axis, ticks and labels
            ax.set_xlabel(xlabel, fontsize=label_dim)
            ax.set_xbound(xbounds[0], xbounds[1])
            ax.set_xlim(xlimits)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontsize=number_dim)

            # set the y axis, ticks and labels
            ax.set_ylabel(ylabel, fontsize=label_dim)
            ax.set_ybound(ybounds[0], ybounds[1])
            ax.set_ylim(ylimits)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=number_dim)

            # set the axis spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_bounds(ybounds[0], ybounds[1])
            ax.spines['bottom'].set_bounds(xbounds[0], xbounds[1])

            # set axis title
            h_ti = ''
            if row_i == 0 and plot_row_label:
                for col_v in np.arange(col_val.shape[1]):
                    h_ti += col_val.columns[col_v] + ': '
                    h_ti += col_val.loc[col_i, col_val.columns[col_v]]
                    h_ti += '' if col_v == col_val.shape[1] - 1 else label_sep
            ax.set_title(h_ti, rotation=col_title_angle, ha=col_title_ha,
                         va=col_title_va, fontsize=label_dim)

            # set the right title if necessary
            st = ''
            if col_i == col_val.shape[0] - 1 and plot_col_label:
                for row_v in np.arange(row_val.shape[1]):
                    st += row_val.columns[row_v] + ': '
                    st += row_val.loc[row_i, row_val.columns[row_v]]
                    st += '' if row_v == row_val.shape[1] - 1 else label_sep
            ax.text(xlimits[-1], ylimits[0] + abs(np.diff(ylimits)) / 2, st,
                    rotation=row_title_angle, fontsize=label_dim,
                    ha=row_title_ha, va=row_title_va)

            # Errorbars
            for idx, y in enumerate(y_var):

                # get the data
                y_avg = []
                y_err_top = []
                y_err_low = []
                y_v = pd.concat([row_df, col_df], 1).to_dict('list')
                for x_v in x_val:
                    y_v[x_var] = [x_v]
                    y_t = myut.extract(data, y_v)[0][y].values
                    y_avg.append(np.mean(y_t))
                    y_err_top.append(ci(y_t)[1] - np.mean(y_t))
                    y_err_low.append(np.mean(y_t) - ci(y_t)[0])
                y_err = [y_err_low, y_err_top]

                # plot the data
                ax.errorbar(x_val, y_avg, yerr=y_err, color=colors[idx],
                            capsize=2, fmt='o', marker='o',
                            markerfacecolor='navy', markeredgecolor='navy',
                            markersize=6)
                if interp_line:
                    line = interpolate(y_avg, len(y_avg) * 100)
                    x = np.linspace(xbounds[0], xbounds[1], len(line))
                    ax.plot(x, line, colors[idx])
    return fig
