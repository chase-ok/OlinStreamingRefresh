
import scaffold
import processing as pr
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


RENDER_MASKS_OUTPUT = scaffold.registerParameter("renderMasksOutput", "masks.avi",
"""The file path to render the output video of RenderForegroundMasks to.""")

class RenderForegroundMasks(_RenderTask):

    name = "Render Foreground Masks"
    dependencies = [pr.ComputeForegroundMasks]

    def run(self):
        images = self._import(pr.ComputeForegroundMasks, "masks")
        self._render(map(im.binaryToGray, images), RENDER_MASKS_OUTPUT)


RENDER_REMOVED_BACKGROUND_OUTPUT = scaffold.registerParameter("renderRemovedBackgroundOutput", "removedBackground.avi",
"""The file path to render the output video of RenderRemovedBackground to.""")

class RenderRemovedBackground(_RenderTask):

    name = "Render Removed Background"
    dependencies = [pr.RemoveBackground]

    def run(self):
        images = self._import(pr.RemoveBackground, "images")
        self._render(images, RENDER_REMOVED_BACKGROUND_OUTPUT)

RENDER_WATERSHED_OUTPUT = scaffold.registerParameter("renderWatershedOutput", "watershed.avi",
"""The file path to render the output video of RenderWatershed to.""")

class RenderWatershed(_RenderTask):

    name = "Render Watershed"
    dependencies = [pr.Watershed]

    def run(self):
        images = self._import(pr.Watershed, "isolated")
        self._render(images, RENDER_WATERSHED_OUTPUT)


RENDER_ELLIPSES_OUTPUT = scaffold.registerParameter("renderEllipsesOutput", "ellipses.avi",
"""The file path to render the output video of RenderEllipses to.""")
RENDER_ELLIPSES_COLOR = scaffold.registerParameter("renderEllipsesColor", (0, 0, 255),
"""The RGB color used to draw the ellipses.""")

class RenderEllipses(_RenderTask):

    name = "Render Ellipses"
    dependencies = [pr.FindEllipses, im.LoadImages]

    def run(self):
        images = self._import(im.LoadImages, "images")
        ellipses = self._import(pr.FindEllipses, "ellipses")
        color = self._param(RENDER_ELLIPSES_COLOR)

        drawn = []
        for image, group in zip(images, pr.groupEllipsesByFrame(ellipses)):
            self.context.debug("Group len={0}", len(group))
            canvas = im.toColor(im.forceRange(image, 0, 255))

            for row in group:
                cv2.ellipse(canvas, 
                            tuple(map(int, ellipses[row]["position"])),
                            tuple(map(lambda x: int(x/2), ellipses[row]["axes"])),
                            ellipses[row]["angle"]*_TO_DEGREES,
                            0, 365,
                            color,
                            1,
                            cv2.CV_AA,
                            0)
            
            drawn.append(canvas)

        self._render(drawn, RENDER_ELLIPSES_OUTPUT)