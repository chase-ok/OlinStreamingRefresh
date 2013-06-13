
import scaffold
import particles as pt
import images as im
import analysis as an
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg

_TO_DEGREES = 180/math.pi

class _RenderTask(scaffold.Task):

    _outputParam = None

    def _render(self, images):
        if self._outputParam is None: raise NotImplemented
        output = self._param(self._outputParam)

        im.ImageSeq(images).writeMovie(output)
        self.context.log("Rendered to {0}.", output)


RENDER_MASKS_OUTPUT = scaffold.registerParameter("renderMasksOutput", "../videos/masks.avi",
"""The file path to render the output video of RenderForegroundMasks to.""")

class RenderForegroundMasks(_RenderTask):

    name = "Render Foreground Masks"
    dependencies = [im.ComputeForegroundMasks]
    _outputParam = RENDER_MASKS_OUTPUT

    def run(self):
        images = self._import(im.ComputeForegroundMasks, "masks")
        self._render(map(im.binaryToGray, images))

RENDER_DIFFS_OUTPUT = scaffold.registerParameter("renderDiffsOutput", "../videos/diffs.avi",
"""The file path to render the output video of RenderDifferences to.""")

class RenderDifferences(_RenderTask):

    name = "Render Differences"
    dependencies = [im.ComputeForegroundMasks]
    _outputParam = RENDER_DIFFS_OUTPUT

    def run(self):
        diffs = self._import(im.ComputeForegroundMasks, "diffs")
        images = [im.forceRange(image*(image > 0), 0, 255).astype(np.uint8) 
                  for image in diffs]
        self._render(images)


RENDER_REMOVED_BACKGROUND_OUTPUT = scaffold.registerParameter("renderRemovedBackgroundOutput", "../videos/removedBackground.avi",
"""The file path to render the output video of RenderRemovedBackground to.""")

class RenderRemovedBackground(_RenderTask):

    name = "Render Removed Background"
    dependencies = [im.RemoveBackground]
    _outputParam = RENDER_REMOVED_BACKGROUND_OUTPUT

    def run(self):
        images = self._import(im.RemoveBackground, "images")
        self._render(images)


RENDER_SEED_IMAGES_OUTPUT = scaffold.registerParameter("renderSeedImagesOutput", "../videos/seedImages.avi",
"""The file path to render the output video of RenderSeedImages to.""")

class RenderSeedImages(_RenderTask):

    name = "Render Seed Images"
    dependencies = [pt.FindParticlesViaEdges]
    _outputParam = RENDER_SEED_IMAGES_OUTPUT

    def run(self):
        images = self._import(pt.FindParticlesViaEdges, "seedImages")
        self._render(images)

RENDER_WATERSHED_OUTPUT = scaffold.registerParameter("renderWatershedOutput", "../videos/watershed.avi",
"""The file path to render the output video of RenderWatershed to.""")

class RenderWatershed(_RenderTask):

    name = "Render Watershed"
    dependencies = [im.Watershed]
    _outputParam = RENDER_WATERSHED_OUTPUT

    def run(self):
        images = self._import(im.Watershed, "isolated")
        self._render(images)


RENDER_REGIONS_OUTPUT = scaffold.registerParameter("renderRegionsOutput", "../videos/regions.avi",
"""The file path to render the output video of RenderRegions to.""")

class RenderRegions(_RenderTask):

    name = "Render Regions"
    dependencies = [im.MergeStatisticalRegions]
    _outputParam = RENDER_REGIONS_OUTPUT

    def run(self):
        images = self._import(im.MergeStatisticalRegions, "masks")
        self._render(map(im.binaryToGray, images))

RENDER_MORPHED_IMAGES_OUTPUT = scaffold.registerParameter("renderMorphedImagesOutput", "../videos/morphImages.avi",
"""The file path to render the output video of RenderMorphImages to.""")

class RenderMorphImages(_RenderTask):

    name = "Render Morph Images"
    dependencies = [pt.FindParticlesViaMorph]
    _outputParam = RENDER_MORPHED_IMAGES_OUTPUT

    def run(self):
        images = self._import(pt.FindParticlesViaMorph, "images")
        self._render(images)

RENDER_ELLIPSES_OUTPUT = scaffold.registerParameter("renderEllipsesOutput", "../videos/ellipses.avi",
"""The file path to render the output video of RenderEllipses to.""")
RENDER_ELLIPSES_COLOR = scaffold.registerParameter("renderEllipsesColor", (0, 0, 255),
"""The RGB color used to draw the ellipses.""")

class RenderEllipses(_RenderTask):

    name = "Render Ellipses"
    dependencies = [pt.FindParticlesViaMorph, im.LoadImages]
    _outputParam = RENDER_ELLIPSES_OUTPUT

    def run(self):
        images = self._import(im.LoadImages, "images")
        ellipses = self._import(pt.FindParticlesViaMorph, "ellipses")
        color = self._param(RENDER_ELLIPSES_COLOR)

        drawn = []
        for image, group in zip(images, pt.groupEllipsesByFrame(ellipses)):
            self.context.debug("Group len={0}", len(group))
            canvas = im.toColor(im.forceRange(image, 0, 255))

            for row in group:
                cv2.ellipse(canvas, 
                            tuple(map(int, ellipses[row]["position"])),
                            tuple(map(int, ellipses[row]["axes"])),
                            ellipses[row]["angle"]*_TO_DEGREES,
                            0, 360,
                            color,
                            1,
                            cv2.CV_AA,
                            0)
            
            drawn.append(canvas)

        self._render(drawn)


class _AnimationTask(_RenderTask):

    _outputParam = None

    def _animate(self, frames, func, **animArgs):
        if self._outputParam is None: raise NotImplemented

        images = []
        figure = plt.Figure(figsize=(6,6), dpi=300, facecolor='w')
        axes = figure.add_subplot(111)
        canvas = FigureCanvasAgg(figure) 

        for frame in frames:
            axes.clear()
            func(frame, axes)
            canvas.draw()
            
            image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8)
            image.shape = canvas.get_width_height() + (3,)
            images.append(image)
        self._render(images)


class _PlotField(_AnimationTask):

    dependencies = [an.GridParticles]
    _fieldType = None # Options are "scalar", "vector"

    def run(self):
        points = self._import(an.GridParticles, "cellCenters")

        def doPlot(row, axes):
            axes.set_title("t = {0}".format(row['time']))

            if self._fieldType == 'scalar':
                shape = self._param(an.NUM_GRID_CELLS)
                axes.pcolormesh(points[:, 0].reshape(shape), 
                                points[:, 1].reshape(shape), 
                                row['data'].reshape(shape))
            elif self._fieldType == 'vector':
                axes.quiver(points[:, 0], points[:, 1], 
                            row['data'][:, 0], row['data'][:, 1])
            else:
                raise NotImplemented
            
        self._animate(self._getTable().iterrows(), doPlot)

    def _getTable(self): raise NotImplemented


PLOT_VELOCITY_FIELD_OUTPUT = scaffold.registerParameter("plotVelocityFieldOutput", "velocityField.avi",
"""The file path to render the output animation of PlotVelocityField to.""")

class PlotVelocityField(_PlotField):

    name = "Plot Velocities"
    dependencies = _PlotField.dependencies + [an.CalculateVelocityField]
    _outputParam = PLOT_VELOCITY_FIELD_OUTPUT
    _fieldType = "vector"

    def _getTable(self): return self._import(an.CalculateVelocityField, "field")


PLOT_DENSITY_FIELD_OUTPUT = scaffold.registerParameter("plotDensityFieldOutput", "densityField.avi",
"""The file path to render the output animation of PlotDensityField to.""")

class PlotDensityField(_PlotField):

    name = "Plot Densities"
    dependencies = _PlotField.dependencies + [an.CalculateDensityField]
    _outputParam = PLOT_DENSITY_FIELD_OUTPUT
    _fieldType = "scalar"

    def _getTable(self): return self._import(an.CalculateDensityField, "field")


class _PlotCorrelation(scaffold.Task):

    def run(self):
        radii = self._param(an.RADII)
        table = self._getTable()

        plt.figure()
        mean = np.empty_like(radii, float)
        for row in table.iterrows():
            plt.plot(radii, row['data'], '.k', markersize=3, label='_nolegend_')
            mean += row['data']

        mean /= table.nrows
        plt.plot(radii, mean, linewidth=3)

        plt.savefig(self._param(self._outputParam), dpi=600)


PLOT_PARTICLE_DISTANCE_OUTPUT = scaffold.registerParameter("plotParticleDistanceOutput", "particleDistance.png",
"""The file path to render the output plot of PlotParticleDistance to.""")

class PlotParticleDistance(_PlotCorrelation):

    name = "Plot Particle Distance"
    dependencies = [an.CorrelateParticleDistance]
    _outputParam = PLOT_PARTICLE_DISTANCE_OUTPUT

    def _getTable(self): 
        return self._import(an.CorrelateParticleDistance, "table")
