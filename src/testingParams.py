
import tables
import scaffold
import processing

def runSeq(name, params):
	with tables.openFile("test.h5", "w") as f:
	    local = f.createGroup("/", "test")
	    d = scaffold.Hdf5Data(f, local, params)
	    d.attrs.configFile = "/home/ckernan/data/streaming/High density/Series080_Properties.xml"
	    t = scaffold.TaskSequence([processing.IdentifyParticles()])
	    t.run(d)

	    d.cache[processing.IdentifiedParticles].writeMovie('/home/ckernan/dev/OlinStreaming2/' + name)

# Still need to test lowThresh and highThresh

for featureRadius in [3]:
	for dilationRadius in [2, 3]:
		for morphThreshold in [0.1, 0.2, 0.3]:
			for blurSigma in [0.1, 0.2, 1.0, 2.0]:
				for expThreshold in [0.001, 0.0001]:
					params = dict(featureRadius=featureRadius,
						          dilationRadius=dilationRadius,
						          morphThreshold=morphThreshold,
						          blurSigma=blurSigma,
						          expThreshold=expThreshold)
					print "Testing params: " + str(params)
					runSeq(str(params) + ".avi", params)

