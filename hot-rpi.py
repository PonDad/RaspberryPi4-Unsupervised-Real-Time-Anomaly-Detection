import time
import datetime
import os
import numpy as np
import random
import math
import subprocess
import warnings

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood 
from htm.bindings.algorithms import Predictor

import matplotlib
import matplotlib.pyplot as plt

dateEncoder = DateEncoder(timeOfDay= (30, 1), weekend  = 21) 

scalarEncoderParams            = RDSE_Parameters()
scalarEncoderParams.size       = 700
scalarEncoderParams.sparsity   = 0.02
scalarEncoderParams.resolution = 0.88
scalarEncoder = RDSE( scalarEncoderParams )
encodingWidth = (dateEncoder.size + scalarEncoder.size)

sp = SpatialPooler(
    inputDimensions            = (encodingWidth,),
    columnDimensions           = (1638,),
    potentialPct               = 0.85,
    potentialRadius            = encodingWidth,
    globalInhibition           = True,
    localAreaDensity           = 0.04395604395604396,
    synPermInactiveDec         = 0.006,
    synPermActiveInc           = 0.04,
    synPermConnected           = 0.13999999999999999,
    boostStrength              = 3.0,
    wrapAround                 = True
)

tm = TemporalMemory(
    columnDimensions          = (1638,), #sp.columnDimensions
    cellsPerColumn            = 13,
    activationThreshold       = 17,
    initialPermanence         = 0.21,
    connectedPermanence       = 0.13999999999999999, #sp.synPermConnected
    minThreshold              = 10,
    maxNewSynapseCount        = 32,
    permanenceIncrement       = 0.1,
    permanenceDecrement       = 0.1,
    predictedSegmentDecrement = 0.0,
    maxSegmentsPerCell        = 128,
    maxSynapsesPerSegment     = 64
)

records=1200

probationaryPeriod = int(math.floor(float(0.1)*records))
learningPeriod     = int(math.floor(probationaryPeriod / 2.0))
anomaly_history = AnomalyLikelihood(learningPeriod= learningPeriod,
                                  estimationSamples= probationaryPeriod - learningPeriod,
                                  reestimationPeriod= 100)

predictor = Predictor( steps=[1, 5], alpha=0.1)
predictor_resolution = 1

inputs      = []
anomaly     = []
anomalyProb = []
predictions = {1: [], 5: []}

plot = plt.figure(figsize=(25,15),dpi=60)
warnings.simplefilter('ignore')

for count in range(records):
    dateObject = datetime.datetime.now()

    cp = subprocess.run(['vcgencmd', 'measure_temp'], encoding='utf-8', stdout=subprocess.PIPE)
    temp = float(cp.stdout[5:-3])
    inputs.append( temp )

    dateBits        = dateEncoder.encode(dateObject)
    tempBits = scalarEncoder.encode(temp)

    encoding = SDR( encodingWidth ).concatenate([tempBits, dateBits])

    activeColumns = SDR( sp.getColumnDimensions() )

    sp.compute(encoding, True, activeColumns)

    tm.compute(activeColumns, learn=True)

    pdf = predictor.infer( tm.getActiveCells() )
    for n in (1,5):
        if pdf[n]:
            predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution )
        else:
            predictions[n].append(float('nan'))
        
    anomalyLikelihood = anomaly_history.anomalyProbability( temp, tm.anomaly )
    
    anomaly.append( tm.anomaly )
    anomalyProb.append( anomalyLikelihood )

    predictor.learn(count, tm.getActiveCells(), int(temp / predictor_resolution))
    print(count," ",dateObject," ",temp)

    time.sleep(5)

plt.subplot(2, 1, 1)
plt.plot(inputs, color='green', linestyle = "solid", linewidth = 2.0,label="Temp")
plt.plot(predictions[1], color='red',linestyle = "dotted",label="Temp Pred Next Step")
plt.ylim(45.0, 65.0)
plt.title("Prediction", fontsize=18)
plt.legend(loc='lower left', fontsize=14)
plt.subplot(2, 1, 2)
plt.plot(anomaly, color='skyblue',linestyle = "dotted",label="Anomaly")
plt.plot(anomalyProb, color='orange', linestyle = "solid", linewidth = 2.0,label="AnomalyProb")
plt.title("Anomaly likelihood", fontsize=18)
plt.legend(loc='lower left', fontsize=14)


plt.savefig('fig.png')
plt.show()