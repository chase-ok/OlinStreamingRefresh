
import scaffold
import particles as pt
import images as im
import numpy as np
import cv2
import math

_TO_DEGREES = 180/math.pi

class _RenderTask(scaffold.Task):

    def _render(self, images, outputParam):
        output = self._param(outputParam)
        im.ImageSeq(images).writeMovie(output)
        self.context.log("Rendered to {0}.", output)


RENDER_MASKS_OUTPUT = scaffold.registerParameter("renderMasksOutput", "../videos/masks.avi",
"""The file path to render the output video of RenderForegroundMasks to.""")

class RenderForegroundMasks(_RenderTask):

    name = "Render Foreground Masks"
    dependencies = [im.ComputeForegroundMasks]

    def run(self):
        images = self._import(im.ComputeForegroundMasks, "masks")
        self._render(map(im.binaryToGray, images), RENDER_MASKS_OUTPUT)

RENDER_DIFFS_OUTPUT = scaffold.registerParameter("renderDiffsOutput", "../videos/diffs.avi",
"""The file path to render the output video of RenderDifferences to.""")

class RenderDifferences(_RenderTask):

    name = "Render Differences"
    dependencies = [im.ComputeForegroundMasks]

    def run(self):
        diffs = self._import(im.ComputeForegroundMasks, "diffs")
        images = [im.forceRange(image*(image > 0), 0, 255).astype(np.uint8) 
                  for image in diffs]
        self._render(images, RENDER_DIFFS_OUTPUT)


RENDER_REMOVED_BACKGROUND_OUTPUT = scaffold.registerParameter("renderRemovedBackgroundOutput", "../videos/removedBackground.avi",
"""The file path to render the output video of RenderRemovedBackground to.""")

class RenderRemovedBackground(_RenderTask):

    name = "Render Removed Background"
    dependencies = [im.RemoveBackground]

    def run(self):
        images = self._import(im.RemoveBackground, "images")
        self._render(images, RENDER_REMOVED_BACKGROUND_OUTPUT)


RENDER_SEED_IMAGES_OUTPUT = scaffold.registerParameter("renderSeedImagesOutput", "../videos/seedImages.avi",
"""The file path to render the output video of RenderSeedImages to.""")

class RenderSeedImages(_RenderTask):

    name = "Render Seed Images"
    dependencies = [pt.FindParticlesViaEdges]

    def run(self):
        images = self._import(pt.FindParticlesViaEdges, "seedImages")
        self._render(images, RENDER_SEED_IMAGES_OUTPUT)

RENDER_WATERSHED_OUTPUT = scaffold.registerParameter("renderWatershedOutput", "../videos/watershed.avi",
"""The file path to render the output video of RenderWatershed to.""")

class RenderWatershed(_RenderTask):

    name = "Render Watershed"
    dependencies = [im.Watershed]

    def run(self):
        images = self._import(im.Watershed, "isolated")
        self._render(images, RENDER_WATERSHED_OUTPUT)


RENDER_REGIONS_OUTPUT = scaffold.registerParameter("renderRegionsOutput", "../videos/regions.avi",
"""The file path to render the output video of RenderRegions to.""")

class RenderRegions(_RenderTask):

    name = "Render Regions"
    dependencies = [im.MergeStatisticalRegions]

    def run(self):
        images = self._import(im.MergeStatisticalRegions, "masks")
        self._render(map(im.binaryToGray, images), RENDER_REGIONS_OUTPUT)


RENDER_ELLIPSES_OUTPUT = scaffold.registerParameter("renderEllipsesOutput", "../videos/ellipses.avi",
"""The file path to render the output video of RenderEllipses to.""")
RENDER_ELLIPSES_COLOR = scaffold.registerParameter("renderEllipsesColor", (0, 0, 255),
"""The RGB color used to draw the ellipses.""")

class RenderEllipses(_RenderTask):

    name = "Render Ellipses"
    dependencies = [pt.FindParticlesViaEdges, im.LoadImages]

    def run(self):
        images = self._import(im.LoadImages, "images")
        ellipses = self._import(pt.FindParticlesViaEdges, "ellipses")
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

        self._render(drawn, RENDER_ELLIPSES_OUTPUT)