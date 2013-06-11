
import tables
import scaffold

def makeContext(**params):
    f = tables.openFile("test2.h5", "a")
    if hasattr(f.root, "data"):
        local = f.root.data
    else:
        local = f.createGroup("/", "data")
    params["configFile"] = "C:\\Users\\chase_000\\SkyDrive\\Research\\Streaming\\Images\\Updated\\Series064_Properties.xml"
    #params["configFile"] = "C:\\Users\\chase_000\\SkyDrive\\Research\\Streaming\\Images\\High\\Series080_Properties.xml"
    #params["configFile"] = "/home/ckernan/data/streaming/One eighth high density/Series154_Properties.xml"
    return scaffold.Context(f, local, params)

if __name__ == "__main__":
    import particles
    import visual
    import analysis
    c = makeContext()
    s = scaffold.Scheduler()
    #s.addTask(visual.RenderSeedImages)
    #s.addTask(visual.RenderDifferences)
    #s.addTask(visual.RenderForegroundMasks)
    #s.addTask(visual.RenderRemovedBackground)
    #s.addTask(visual.RenderWatershed)
    #s.addTask(visual.RenderRegions)
    s.addTask(analysis.CalculateLocalVelocityCorrelationField)
    s.addTask(analysis.CorrelateParticleDistance)
    s.run(c)
    c.hdf5.close()
