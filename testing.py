
import tables
import scaffold

def makeContext(**params):
    f = tables.openFile("test.h5", "w")
    local = f.createGroup("/", "test")
    params["configFile"] = "C:\\Users\\ckernan\\SkyDrive\\Research\\Streaming\\Images\\High\\Series080_Properties.xml"
    #params["configFile"] = "/home/ckernan/data/streaming/One eighth high density/Series154_Properties.xml"
    return scaffold.Context(f, local, params)

if __name__ == "__main__":
    import processing
    import visual
    c = makeContext()
    s = scaffold.Scheduler()
    #s.addTask(visual.RenderForegroundMasks)
    #s.addTask(visual.RenderRemovedBackground)
    #s.addTask(visual.RenderWatershed)
    s.addTask(visual.RenderEllipses)
    s.addTask(visual.RenderRegions)
    s.run(c)
