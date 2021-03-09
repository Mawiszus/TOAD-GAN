from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d, Axes3D
from matplotlib import offsetbox
import numpy as np


class ImageAnnotations3D:
    def __init__(self, xyz, imgs: List[np.ndarray], labels: List[str], ax3d: Axes3D, figure: Figure):
        self.xyz = xyz
        self.imgs = imgs
        self.labels = labels
        self.ax3d = ax3d
        self.figure = figure
        self.annot = []
        for xyz, im, label in zip(self.xyz, self.imgs, self.labels):
            x, y = self.proj(xyz)
            self.annot.append(self.image(im, [x, y]))
            self.annot.append(self.label(label, [x, y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect(
            "draw_event", self.update)

        self.funcmap = {"button_press_event": self.ax3d._button_press,
                        "motion_notify_event": self.ax3d._on_move,
                        "button_release_event": self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb)
                    for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1, 
            calculate position in 2D in ax2 """
        x, y, z = X
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax3d.get_proj())
        return x2, y2

    def image(self, arr, xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr)
        ab = offsetbox.AnnotationBbox(im, xy, pad=0)
        self.ax3d.add_artist(ab)
        return ab

    def label(self, label, xy):
        text = offsetbox.TextArea(label, minimumdescent=False)
        ab = offsetbox.AnnotationBbox(text, xy,
                                      xybox=(0, 16),
                                      xycoords='data',
                                      boxcoords="offset points")
        self.ax3d.add_artist(ab)
        return ab

    def update(self, event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s, ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)
