
import tables
import scaffold
import images
import particles
import visual
import analysis

def makeContext(**params):
    f = tables.openFile("testNew.h5", "a")
    params["configFile"] = "C:\\Users\\chase_000\\SkyDrive\\Research\\Streaming\\Images\\Updated\\Series064_Properties.xml"
    #params["configFile"] = "C:\\Users\\chase_000\\SkyDrive\\Research\\Streaming\\Images\\High\\Series080_Properties.xml"
    #params["configFile"] = "/home/ckernan/data/streaming/One eighth high density/Series154_Properties.xml"
    return scaffold.Context(f, f.root, params)

def extractId(context):
    scaffold.Scheduler([images.ExtractUniqueId]).run(context)

def doAll(context):
    s = scaffold.Scheduler()
    s.addTask(images.ComputeForegroundMasks)
    s.addTask(images.RemoveBackground)
    #s.addTask(images.Watershed)
    s.addTask(visual.RenderSeedImages)
    #s.addTask(visual.RenderDifferences)
    #s.addTask(visual.RenderForegroundMasks)
    #s.addTask(visual.RenderRemovedBackground)
    #s.addTask(visual.RenderWatershed)
    #s.addTask(visual.RenderRegions)
    s.addTask(analysis.CalculateDensityField)
    s.addTask(analysis.CalculateVelocityField)
    s.addTask(analysis.CalculateLocalVelocityCorrelationField)
    s.addTask(analysis.CorrelateParticleDistance)
    s.addTask(analysis.CorrelateDirectorVelocities)
    s.addTask(analysis.CorrelateDirectors)
    s.addTask(analysis.CorrelateVelocities)
    #s.addTask(visual.PlotVelocityField)
    #s.addTask(visual.PlotDensityField)
    s.addTask(visual.PlotParticleDistance)
    s.run(context, forceRedo=False)

if __name__ == "__main__":
    c = makeContext()
    try:
        extractId(c)
        doAll(c)
    finally:
        c.hdf5.close()
