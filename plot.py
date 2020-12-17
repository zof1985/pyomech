


from bokeh.models import Label
from bokeh.plotting import figure, show
from bokeh.util.compiler import TypeScript

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

    let sx = this.model.x_units == "data" ? this.coordinates.x_scale.compute(this.model.x) : panel.xview.compute(this.model.x)
    let sy = this.model.y_units == "data" ? this.coordinates.y_scale.compute(this.model.y) : panel.yview.compute(this.model.y)

    sx += this.model.x_offset
    sy -= this.model.y_offset

    //--- End of copied section from ``Label.render`` implementation
    // Must render as superpositioned div (not on canvas) so that KaTex
    // css can properly style the text
    this._css_text(this.layer.ctx, "", sx, sy, angle)

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
  __view_type__: LatexLabelView

  constructor(attrs?: Partial<LatexLabel.Attrs>) {
    super(attrs)
  }

  static init_LatexLabel() {
    this.prototype.default_view = LatexLabelView
  }
}
"""


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
    __implementation__ = TypeScript(TS_CODE)


'''
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
    """
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
    """

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
    """
    generate a 3D axis where to plot data
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pl
    return pl.axes(projection='3d')
'''