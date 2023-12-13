import os, re, struct, glob, copy, collections.abc, biosppy, time
import pandas as pd, datetime as dt, numpy as np
import matplotlib.pyplot as plt, matplotlib.dates as md
from scipy import signal
from scipy.stats import describe, moment
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

# A class to represent a pretty float to 2 decimal places.
class PrettyFloat2(float):
    def __repr__(self):
        return "%0.2f" % self

# A class to represent a pretty float to 4 decimal places.
class PrettyFloat4(float):
    def __repr__(self):
        return "%0.4f" % self

# A class to represent a pretty float to 6 decimal places.
class PrettyFloat6(float):
    def __repr__(self):
        return "%0.6f" % self

class ConvenienceFunctions(object):
    def __init__(self):
        pass
    def ReadJsonAsPdAndFilter(jsonPath, filterList=[]):
        '''
            Reads the json file and returns a pandas dataframe with the filtered columns

            Parameters
            ----------
            jsonPath : str
                Path to the json file
            filterList : list of tuples
                List of properties to filter the filenames by
                Example: [('lowNoise','True'), ('site','SiteY'), ('subjectType','LVO')]

            Returns
            -------
            pdData : pandas dataframe
                Pandas dataframe with the filtered columns
        '''
        pdData = pd.read_json(jsonPath)
        for filter in filterList:
            pdData = pdData[pdData[filter[0]] == filter[1]]
        pdData = pdData.reset_index(drop=True)
        return pdData
    
    def CleanName(name):
        '''
            Cleans the name of the file by converting to upper case and removing the leading zeros in any number

            Parameters
            ----------
            name : str
                Name of the file

            Returns
            -------
            name : str
                Cleaned name of the file
        '''
        #Split the name into parts aphabet and number
        parts = re.split('(\d+)', name)
        #Check if the first part is a number
        if parts[0].isdigit():
            return name.lstrip('0')
        return parts[0].upper() + parts[1].lstrip('0')
    
    def NormalizeFeaturesAndRunKFoldCrossValidationWithRF(x, y, folds=5, RFDepth=3, featureNames=None):
        '''
            Normalizes the features and runs k-fold cross validation with RF

            Parameters
            ----------
            x : numpy array
                Features
            y : numpy array
                Labels
            folds : int
                Number of folds for k-fold cross validation
            RFDepth : int
                Depth of the random forest
            featureNames : list of str
                Names of the features

            Returns
            -------
            mean_auc : float
                Mean AUC of the ROC curve
        '''
        sc = StandardScaler()
        X_scaled = sc.fit_transform(x)

        if featureNames is not None:
            if RFDepth > 0:
                clf = RandomForestClassifier(max_depth=RFDepth)
            else:
                clf = RandomForestClassifier()
            clf.fit(X_scaled, y.ravel()) 
            #Get feature importance sorted by importance
            featureImportance = np.argsort(clf.feature_importances_)[::-1]
            #Print feature importance and names on the same line with 4 decimal places and stop at 5 features
            for i in range(len(featureImportance)):
                print(featureNames[featureImportance[i]], PrettyFloat4(clf.feature_importances_[featureImportance[i]]),featureImportance[i], end=", ")
                if i == 4:
                    break
            print()

        #Set a seed for reproducibility(should be random otherwise)
        np.random.seed(454545)

        #Create ROC curve with RF using k-fold cross validation
        cv = StratifiedKFold(n_splits=folds)
        if RFDepth > 0:
            classifier = RandomForestClassifier(max_depth=RFDepth)
        else:
            classifier = RandomForestClassifier()

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X_scaled, y.ravel())):
            classifier.fit(X_scaled[train], y[train].ravel())
            viz = RocCurveDisplay.from_estimator(
                classifier,
                X_scaled[test],
                y[test].ravel(),
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        plt.show()
        return mean_auc, (mean_fpr, mean_tpr)
    
class ScanParams(object):
    """Class to store the scan parameters - device parameters, physical constant, processing flags"""
    
    def __init__(self, scanPath, calPathIn = None, scanTypeIn = 2, correctionTypeIn = -1):
        """
            Creates a parameter object of HeadScanParams class
            
            Parameters
            ----------
            scanName : str
                Folder on disk where the scan data is stored
            calPathIn : str
                Folder to the disk where the calibration data is stored
            scanTypeIn : str
                0 - Unilateral, 1 - Bilateral, 2 - Long scan,  3 - Four Camera, 4 - Four Camera long scan
                Pattern used to scan subject
        """
        #TODO: Move the general scan params into this class
        self.path = scanPath
        self.scanType = scanTypeIn-1
        #0 - Initial device scan order Right Horizontal, Left Horizontal, Right Temple, Left Temple
        #1 - Simultaneous scan order Horizontal, Near, Temple, Horizontal Repeat
        #All processing is done Ch0 and then Ch1
        self.cameraGain = [[16,16,16,16,1,1,1,1],[16,1,16,16,16,1,16,16]] #1 is high
        self.cameraPosition = [['RH','LH','RV','LV','RN','LN','RN','LN',],
                               ['RH','LN','RV','RH','LH','RN','LV','LH',]]
        self.correctionType = correctionTypeIn
        self.printStats = False
        self.imageWidth = 2320
        self.histLength  = 1028 #Includes the last four numbers added on
        self.numBinsHist = 1024
        self.darkBinThresh = [256,128]
        self.hiGainSetting = []
        self.noisyBinMin = 100
        self.dt = 0.025

class PulseFeatures(object):
    """
    Class to store the pulse features
    
    Parameters
    ----------
    samplingIn : float
        Sampling rate of the input data

    """
    def __init__(self,samplingIn=1/40):
        self.sampling       = samplingIn
        self.areaUnderCurve = None
        self.areaUnderCurveP1 = None
        self.amplitude      = None
        self.average        = None
        self.unbiasedAmp    = None
        self.modulationDepth= None
        self.skewness       = None
        self.kurtosis       = None
        self.pulseCanopy    = None
        self.pulseOnset     = None
        self.pulseOnsetProp = None
        self.secondMoment   = None
        self.veloCurveIndex = None
        self.veloCurveIndexHann = None
        self.veloCurveIndexNorm = None
        self.veloCurveIndexHannNorm = None
        self.noiseMetric = None
        self.featureList = ['modulationDepth','areaUnderCurve','areaUnderCurveP1','skewness','kurtosis','pulseCanopy','pulseOnset','pulseOnsetProp','secondMoment','amplitude','unbiasedAmp','veloCurveIndex','veloCurveIndexHann',
                            'veloCurveIndexNorm','veloCurveIndexHannNorm','noiseMetric']
        self.featureNames = ['Modulation depth','Area under curve','Area under curve P1','Skewness','Kurtosis',
                            'Pulse canopy','Pulse onset','Pulse onset proportion','Second Moment','Amplitude','Unbiased Amplitude','Velocity curve index',
                            'Velocity curve index Hanning','Velocity curve index normalized','Velocity curve index Hanning normalized','noiseMetric']
        self.featruesAndNames = dict(zip(self.featureList,self.featureNames))

    def GetAreaUnderCurve(self, goldenPulse):
        '''
            Returns the area under the curve for the input waveform

            Parameters
            ----------
            goldenPulse : 1D numpy array
                Input goldenPulse

            Returns
            -------
            areaUnderCurve : float
                Area under the curve
        '''
        mini = np.amin(goldenPulse)
        goldenPulse -= mini
        maxi = np.amax(goldenPulse)
        goldenPulse /= maxi
        
        #Smooth data for better peak detection
        #100 ms window is the length we are shooting for
        window = round(0.1/(self.sampling))
        smoothPulse = signal.convolve(goldenPulse,
                                        signal.hann(window)/np.sum(signal.hann(window)),
                                        'same')
        argmax = np.argmax(smoothPulse)
        return np.sum(goldenPulse), np.sum(smoothPulse[:argmax])

    def ComputeVCI(self, goldenPulseIn, hanningFilter=False):
        '''
            Computes the velocity curve index for the input waveform

            Parameters
            ----------
            goldenPulseIn : 1D numpy array
                Input goldenPulse
            hanningFilter : bool
                If true, apply a hanning filter to the input waveform

            Returns
            -------
            canopyVCI : float
                Velocity curve index for the canopy
            pulseLengthNormalizedCanopyVCI : float
                Velocity curve index for the canopy normalized by the pulse length
        '''
        #Strech pulse between 0-1
        goldenPulse = copy.deepcopy(goldenPulseIn)
        goldenPulse -= np.amin(goldenPulse)
        goldenPulse /= np.amax(goldenPulse)
        goldenPulse = 1-goldenPulse
        gPIndex = np.arange(len(goldenPulse))*self.sampling
        #We use 4x oversampling to get a smoother gradient
        gPIndexUpSamp = np.arange(len(goldenPulse)*4)*self.sampling/4
        gPUpSample = np.interp(gPIndexUpSamp, gPIndex, goldenPulse)
        if hanningFilter:
            #90 ms window is the length we are shooting for
            window = round(0.09/(self.sampling/4))
            gPUpSample = signal.convolve(gPUpSample,
                                         signal.hann(window)/np.sum(signal.hann(window)),
                                         'same')
        grad  = np.gradient(gPUpSample,self.sampling/4)
        grad2 = np.gradient(grad,self.sampling/4)
        k = np.abs(grad2)/((1+grad**2)**1.5)
        indsInCanopy = gPUpSample > 0.25
        indsInCanopy = np.flatnonzero(indsInCanopy)
        canopyVCI = np.sum(k[indsInCanopy])
        return canopyVCI, canopyVCI/len(gPIndexUpSamp)

    def ComputeWaveformAttributes(self, goldenPulse):
        '''
            Computes the following features:
            Canopy - the proportion of time spent above 25% of the systolic-diastolic range
            Onset  - the time taken to reach systolic maximum from onset 0-90% ranges are used to filter out fit/averaging noise
            Relative onset - time spent in systolic part relative to whole pulse

            Parameters
            ----------
            goldenPulse : 1D numpy array
                Fitted/average pulses in the scan

            Returns
            -------
            canopy : float
                Proportion of time spent above 25% of the systolic-diastolic range
            onset : float
                Time taken to reach systolic maximum from onset 0-90% ranges are used to filter out fit/averaging noise
            onsetProp : float
                Time spent in systolic part relative to whole pulse
            variance : float
                Variance of the pulse
            velCurInd : float
                Velocity curve index for the canopy
            velCurIndNorm : float
                Velocity curve index for the canopy normalized for the pulse length
            velCurIndHann : float
                Velocity curve index for the canopy with a hanning filter
            velCurIndHannNorm : float
                Velocity curve index for the canopy with a hanning filter normalized for the pulse length
        '''
        pulseRange = np.nanmax(goldenPulse)-np.nanmin(goldenPulse)
        totalTime  = np.count_nonzero(~np.isnan(goldenPulse))
        indsInCanopy = goldenPulse < np.nanmax(goldenPulse)-pulseRange*0.25
        velCurInd, velCurIndNorm = self.ComputeVCI(goldenPulse)
        velCurIndHann, velCurIndHannNorm = self.ComputeVCI(goldenPulse,True)
        canopy    = np.sum(indsInCanopy)/totalTime
        startInd  = np.nanargmax(goldenPulse[:5])
        sysRange  = np.nanmax(goldenPulse)-pulseRange*0.9
        sysInds   = goldenPulse>sysRange
        endInd    = np.nonzero(~sysInds)[0][0]
        onset     = float(abs(endInd-startInd))*self.sampling
        onsetProp = float(abs(endInd-startInd))/totalTime
        return canopy, onset, onsetProp, moment(goldenPulse,2), velCurInd, velCurIndHann, velCurIndNorm, velCurIndHannNorm
    
    def GetWaveformAttributesForSingleChannelPulse(self,goldenPulse):
        '''
            Computes the following features:
            Area under curve
            Area under curve for the systolic part
            Amplitude
            Modulation depth
            Skewness
            Kurtosis
            Canopy - the proportion of time spent above 25% of the systolic-diastolic range
            Onset  - the time taken to reach systolic maximum from onset 0-90% ranges are used to filter out fit/averaging noise
            Relative onset - time spent in systolic part relative to whole pulse

            Parameters
            ----------
            goldenPulse : 1D numpy array
                Fitted/average pulses in the scan
        '''
        if np.isnan(goldenPulse).all():
            return
        gpDistribution = goldenPulse-np.amin(goldenPulse)
        gpDistribution = np.amax(gpDistribution)-gpDistribution
        gpDistribution = gpDistribution / np.sum(gpDistribution)
        self.areaUnderCurve, self.areaUnderCurveP1 = self.GetAreaUnderCurve(np.copy(gpDistribution))
        self.amplitude = np.amax(goldenPulse) - np.amin(goldenPulse)
        self.modulationDepth = self.amplitude/np.mean(goldenPulse)
        gpDescription  = describe(gpDistribution)
        self.skewness = gpDescription.skewness
        self.kurtosis = gpDescription.kurtosis
        canopy, onset, onsetProp, secMoment, vci, vciHann, vciNorm, vciHannNorm = \
                self.ComputeWaveformAttributes(goldenPulse)
        self.pulseCanopy    = canopy
        self.pulseOnset     = onset
        self.pulseOnsetProp = onsetProp
        self.secondMoment   = secMoment
        self.veloCurveIndex = vci
        self.veloCurveIndexHann = vciHann
        self.veloCurveIndexNorm = vciNorm
        self.veloCurveIndexHannNorm = vciHannNorm
    
    def AppendWaveformAttributesForSingleChannelPulse(self,pulseIn):
        '''
            Computes the following features:
            Area under curve
            Area under curve for the systolic part
            Amplitude
            Modulation depth
            Skewness
            Kurtosis
            Canopy - the proportion of time spent above 25% of the systolic-diastolic range
            Onset  - the time taken to reach systolic maximum from onset 0-90% ranges are used to filter out fit/averaging noise
            Relative onset - time spent in systolic part relative to whole pulse

            Parameters
            ----------
            goldenPulse : 1D numpy array
                Fitted/average pulses in the scan
        '''
        pulse = np.copy(pulseIn)
        #If this is the first pulse, then initialize the arrays
        if self.areaUnderCurve is None:
            for feature in self.featureList:
                self.__dict__[feature] = []
        #Drop leading nan values
        while np.isnan(pulse[0]):
            pulse = pulse[1:]
        #Drop trailing nan values
        while np.isnan(pulse[-1]):
            pulse = pulse[:-1]
        # if any of the remaining values are nan, then skip this pulse
        if np.isnan(pulse).any():
            for feature in self.featureList:
                self.__dict__[feature].append(np.nan)
            return
        
        gpDistribution = pulse-np.amin(pulse)
        gpDistribution = np.amax(gpDistribution)-gpDistribution
        gpDistribution = gpDistribution / np.sum(gpDistribution)
        areaUnderCurve, areaUnderCurveP1 = self.GetAreaUnderCurve(np.copy(gpDistribution))
        self.areaUnderCurve.append(areaUnderCurve)
        self.areaUnderCurveP1.append(areaUnderCurveP1)
        self.amplitude.append(np.amax(pulse) - np.amin(pulse))
        self.average.append(np.mean(pulse))
        self.unbiasedAmp.append(np.amax(pulse[:int(pulse.shape[0]/2)]) - np.amin(pulse[:int(pulse.shape[0]/2)]))
        self.modulationDepth.append( self.amplitude/np.mean(pulse) )
        gpDescription  = describe(gpDistribution)
        self.skewness.append( gpDescription.skewness )
        self.kurtosis.append( gpDescription.kurtosis )
        canopy, onset, onsetProp, secMoment, vci, vciHann, vciNorm, vciHannNorm = \
                self.ComputeWaveformAttributes(pulse)
        self.pulseCanopy.append(canopy)
        self.pulseOnset.append(onset)
        self.pulseOnsetProp.append(onsetProp)
        self.secondMoment.append(secMoment)
        self.veloCurveIndex.append(vci)
        self.veloCurveIndexHann.append(vciHann)
        self.veloCurveIndexNorm.append(vciNorm)
        self.veloCurveIndexHannNorm.append(vciHannNorm)
        slopeChangeCount = np.absolute(np.diff(np.heaviside(np.diff(pulse,n=1),1))).sum()
        self.noiseMetric.append(slopeChangeCount)
    
class ChannelData(object):
    '''
    Class to hold data for a single channel
    '''
    def __init__(self):
        self.imagePathLaserOff = None
        self.imageLaserOffHistWidth = None
        self.imageLaserOffImgMean = None
        self.imagePathLaserOn  = None
        self.histogramPath     = None
        self.darkHistogramPath = None
        self.dataAvailable     = True
        self.contrastNoFilter  = None
        self.correctedMean     = None
        self.contrast  = None
        self.imageMean = None
        self.imageStd  = None
        self.camTemps  = None
        self.hr        = None
        self.initialCamTemp = None
        self.timeStamps = None
        self.NCCPulse  = None
        self.goldenPulse = None
        self.pulseSegments = None
        self.onsets = None
        self.pulseValid = None
        self.splinePulse = None
        self.channelPosition = None
        self.goldenPulseFeatures = None
        self.pulseSegmentsFeatures = None
        
        self.contrastVertNorm = None
        self.imgMeanVertNorm = None

class ReadGen2Data:
    '''
    Class to read in data from Gen2 camera
    '''
    def __init__(self, pathIn, deviceID=-1, correctionTypeIn=0, scanTypeIn=1, enablePlotsIn=False, filterDriftIn=False,
                 filterNoiseIn=True):
        self.path = pathIn
        #Check if dark hitogram is present
        tmpList = glob.glob(os.path.join(self.path,"histo_output_darkscan_ch_*"))
        if len(tmpList)==0:
            self.darkHistogramsAvailable = False
        else:
            self.darkHistogramsAvailable = True
        #Set 4 channel scan and long scan types automagically
        if 'LONGSCAN_4C' in self.path:
            scanTypeIn = 4
            correctionTypeIn = 1
            print('Long scan with 4 channels detected. Setting scanType to {} and correctionType to {}.'.format(scanTypeIn,correctionTypeIn))
        elif 'LONGSCAN' in self.path:
            scanTypeIn = 2
            correctionTypeIn = 1
            print('Long scan detected. Setting scanType to {} and correctionType to {}.'.format(scanTypeIn,correctionTypeIn))
        elif 'FULLSCAN_4C' in self.path:
            scanTypeIn = 3
            correctionTypeIn = 1
            print('4 channel scan detected. Setting scanType to {} and correctionType to {}.'.format(scanTypeIn,correctionTypeIn))
        elif 'FULLSCAN' in self.path and self.darkHistogramsAvailable:
            scanTypeIn = 1
            correctionTypeIn = 1
            print('Simultaneous scan detected. Setting scanType to {} and correctionType to {}.'.format(scanTypeIn,correctionTypeIn))
        if not self.darkHistogramsAvailable:
            if scanTypeIn==2:
                correctionTypeIn = 2
            else:
                correctionTypeIn = 0
            print('Dark histograms not available. Setting correctionType to {} for scanType {}.'.format(correctionTypeIn,scanTypeIn))

        self.scanType = scanTypeIn
        #0 - Initial device scan order Right Horizontal, Left Horizontal, Right Temple, Left Temple
        #1 - Simultaneous scan order Horizontal, Near, Temple, Horizontal Repeat
        #2 - Long scan 
        #3 - 4 channel scan
        #4 - Long scan with 4 channels
        #All processing is done Ch0 and then Ch1
        self.cameraGain = [[16,16,16,16,1,1,1,1], #Initial unilateral scan
                           [16,1,16,16,16,1,16,16], #Simultaneous scan
                           [16,16], #Long scan
                           [16,16,1,16,
                            1,16,16,1,
                            16,16,1,16,
                            1,16,16,1], #4 channel scan
                            [16,1,16,1], #Long scan with 4 channels
                            ]
        self.cameraPosition = [['RH','LH','RV','LV','RN','LN','RN','LN',], #Initial unilateral scan
                               ['RH','LN','RV','RH','LH','RN','LV','LH',], #Simultaneous scan
                               ['RH','LH'],                               #Long scan
                               ['RH','RH','LN','RH',
                                'LN','RV','RV','LN',
                                'LH','LH','RN','LH',
                                'RN','LV','LV','RN',], #4 channel scan
                                ['RH','LN','LH','RN'], #Long scan with 4 channels
                               ]
        self.correctionType = correctionTypeIn
        self.longScanCorrectionFactor = None
        self.printStats = False
        self.enablePlots = enablePlotsIn
        self.filterDrift = filterDriftIn
        self.filterNoise = filterNoiseIn
        self.imageWidth = 2320
        self.histLength  = 1028 #Includes the last four numbers added on
        self.numBinsHist = 1024
        self.darkBinThresh = [256,128]
        self.hiGainSetting = []
        self.noisyBinMin = 100
        self.ADCgain = 0.12 # photons/electrons
        self.dt = 0.025
        self.minbpm = 30
        self.maxbpm = 180
        self.deviceID = deviceID
        self.debugMode = 1 # 0: default. 1: also loads dark images, saves full channel outputs

        if scanTypeIn==2 or scanTypeIn==4:
            #Get number of histograms in the folder
            numScans = len(glob.glob1(self.path,"*histo_output_longscan_*"))
            self.channelData = [ ChannelData() for i in range(numScans) ]
        else:
            if self.scanType<2:
                self.channelData = [ ChannelData() for i in range(8) ]
            else:
                self.channelData = [ ChannelData() for i in range(16) ]

    def printTimestamp(self, timestamp):
        time_sec = timestamp // 1000000000
        time_ns = timestamp % 1000000000
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_sec))
        time_str += '.{:03d}'.format(time_ns // 1000000)
        print(time_str)

    def GetTimestamps(self, tsBytes):
        '''
            Returns the timestamps for the histogram bins

            Parameters
            ----------
            tsBytes : 2D numpy array
                Histogram bins

            Returns
            -------
            timestamps : 1D list
                Timestamps for the histogram bins
        '''
        #Check if all the input bins are zero
        if np.all(tsBytes==0):
            return None
        timestamps = []
        for i in range(tsBytes.shape[0]):
            if tsBytes[i,0]==0 and tsBytes[i,1]==0:
                break
            lowerTS = tsBytes[i,0]
            upperTS  = tsBytes[i,1]
            timestamps.append( int(upperTS)<<32 | int(lowerTS)  )
        return timestamps
    
    def ReadHistogram(self, histogramPath):
        '''
            Reads the histogram file and returns the histogram data, object mean, object std, camera temperature and timestamps
        '''
        histogramFile = open(histogramPath, 'rb')
        histogramData1 = histogramFile.read()
        histogramFile.close()
        
        histogramu32 = np.frombuffer(histogramData1, dtype=np.uint32)
        histogramData = np.copy(histogramu32)
        histogramData = histogramData.reshape((histogramu32.shape[0]//(self.numBinsHist+4),self.numBinsHist+4))
        timeStamps    = self.GetTimestamps(histogramData[:,14:16])
        histogramData = histogramData[:,:-4]
        histogramData[:,14:18] = 0
        
        histogramf32 = np.frombuffer(histogramData1, dtype=np.float32)
        histogramTemp = np.copy(histogramf32)
        histogramTemp = histogramTemp.reshape((histogramf32.shape[0]//(self.numBinsHist+4),self.numBinsHist+4))
        obMean = np.copy(histogramTemp[:,16])
        obStd  = np.copy(histogramTemp[:,17])

        histogramu8 = np.frombuffer(histogramData1, dtype=np.uint8)
        histogramTemp = np.copy(histogramu8)
        histogramTemp = histogramTemp.reshape((int(histogramu8.shape[0]/4)//(self.histLength),self.histLength*4))
        camTemps = np.flip(histogramTemp[:,4104:4108],axis=1)
        camInd = int(re.split('scan_ch_', histogramPath)[1][0])
        if(self.scanType <3): 
            camTemps[:,[1,3]] = camTemps[:,[1,3]] * 1.5625 - 45 # full camera temperature data output
            camTemps = camTemps[:,camInd*2+1] # selecting single camera's temp
        else:
            camTemps[:,camInd] = camTemps[:,camInd] * 1.5625 - 45 # full camera temperature data output
            camTemps = camTemps[:,camInd] # selecting single camera's temp
        
        return histogramData, obMean, obStd, camTemps, timeStamps
    
    def GetHistogramStats(self, hist, bins):
        binsSq = np.multiply(bins,bins)
        if hist.ndim==2:
            mean = np.zeros(hist.shape[0])
            std  = np.zeros(hist.shape[0])
            histWid = np.zeros(hist.shape[0])
            for i in range(hist.shape[0]):
                hist[i][hist[i]<self.noisyBinMin] = 0
                mean[i] = np.dot(hist[i],bins)/np.sum(hist[i])
                var = (np.dot(hist[i],binsSq)-mean[i]*mean[i]*np.sum(hist[i]))/(np.sum(hist[i])-1)
                std[i] = np.sqrt(var)
                histWid[i] = np.sum(hist[i]>100)
        else:
            hist[hist<self.noisyBinMin] = 0
            mean = np.dot(hist,bins)/np.sum(hist)
            var = (np.dot(hist,binsSq)-mean*mean*np.sum(hist))/(np.sum(hist)-1)
            std = np.sqrt(var)
            histWid = np.sum(hist>100)
        
        return mean, std, histWid

    def GetImageStats(self, image):
        histDark, bins = np.histogram(image[:20,:], bins=list(range(self.numBinsHist)))
        histDark = histDark[:-1]; bins = bins[:-2] #Remove the 1023 bin
        obMean, obStd, obWidth = self.GetHistogramStats(histDark, bins)

        histExp, bins = np.histogram(image[20:,:], bins=list(range(self.numBinsHist)))
        histExp = histExp[:-1]; bins = bins[:-2]
        expRowsMean, expRowsStd, expRowsWidth = self.GetHistogramStats(histExp, bins)

        return expRowsMean, expRowsStd, expRowsWidth, obMean, obStd, obWidth

    def ReadImage(self, imagePath):
        imageFile = open(imagePath, 'rb')
        imageData = imageFile.read()
        imageFile.close()
        if len(imageData)%self.imageWidth:
            print('Incomplete image file. Skipping',imagePath)
            imageData = np.array([])
        else:
            imageData = np.array(struct.unpack('<'+str(int(len(imageData)/2))+'H',imageData)).reshape((int(len(imageData)/(self.imageWidth*2)),self.imageWidth))
        return imageData

    def natural_sort(self,l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def CheckFolderForHistogramsAndImages(self, chPos):
        if self.scanType==2 or self.scanType==4:
            histogramPatternCh = os.path.join( self.path, "histo_output_longscan_ch_{}*.bin".format(chPos) )
        else:
            histogramPatternCh = os.path.join( self.path, "histo_output_fullscan_ch_{}*.bin".format(chPos) )
        chHistFiles = sorted(glob.glob(histogramPatternCh))

        darkHistogramPatternCh = os.path.join( self.path, "histo_output_darkscan_ch_{}*.bin".format(chPos) )
        chDarkHistFiles = sorted(glob.glob(darkHistogramPatternCh))
        
        imagePatternCh = os.path.join( self.path, "csix_raw_output_ch_{}*_exp0_*x*_10bit_bayer.y".format(chPos) )
        chImageFiles = self.natural_sort(glob.glob(imagePatternCh))

        if self.scanType==2 or self.scanType==4 or len(chImageFiles)==8: #Long scans can have arbitrary number of images
            chImageFiles = chImageFiles
        elif len(chImageFiles)==16: #When we have two or more images for laser on/off. Pick every other one.
            skipNum = len(chImageFiles)%8
            chImageFiles = [chImageFiles[i] for i in range(len(chImageFiles)) if i % 2 == skipNum]
        else:
            chImageFiles = []
            errorString = 'Missing image files for channel '+str(chPos)+' in the folder:'+self.path
            raise ValueError(errorString)

        return chHistFiles, chImageFiles, chDarkHistFiles

    def ReadHistogramAndImageFileNames(self):
        if self.scanType < 3:
            numChannels = 2
        else:
            numChannels = 4
        if self.scanType==2 or self.scanType==4:
            numScansPerChannel = len(glob.glob1(self.path,"*histo_output_longscan_ch_0_*"))
        else:
            numScansPerChannel = 4
        for i in range(numChannels):
            chHistPaths, chImages, chDarkHistPaths = self.CheckFolderForHistogramsAndImages(i)

            if (not(self.scanType!=2 or self.scanType!=4)) and len(chHistPaths)!=4: #Long scans can have arbitrary number of histograms
                errorString = 'Missing histogram files for ch '+' in the folder:'+self.path
                raise ValueError(errorString)

            for ind, histFileName in enumerate(chHistPaths):
                self.channelData[i*numScansPerChannel+ind].histogramPath = histFileName

            for ind in range(0, len(chImages), 2):
                self.channelData[i*numScansPerChannel+int(ind/2)].imagePathLaserOff = chImages[ind]
                self.channelData[i*numScansPerChannel+int(ind/2)].imagePathLaserOn  = chImages[ind+1]

            for ind, histFileName in enumerate(chDarkHistPaths):
                self.channelData[i*numScansPerChannel+ind].darkHistogramPath = histFileName
        
        if 0: #Debugging filenames
            for ch in self.channelData:
                print(os.path.split(ch.histogramPath)[1],' Lsr off:', os.path.split(ch.imagePathLaserOff)[1],
                      ' Lsr on:', os.path.split(ch.imagePathLaserOn)[1],
                      ' Dark Hist:', os.path.split(ch.darkHistogramPath)[1])
        return

    def computeCorrectionForLongScansWithNoDarkHists(self,chPos):
        chStrs = ['ch_0','ch_1']
        chStr = chStrs[chPos]
        chInd = [i for i, x in enumerate(self.channelData) if x.imagePathLaserOff.find(chStr) != -1]
        imageMainRowsMean = np.zeros((len(chInd)+1,))
        imageMainRowsStd  = np.zeros((len(chInd)+1,))
        for ind, i in enumerate(chInd):
            imageMain = self.ReadImage(self.channelData[i].imagePathLaserOff)
            imageMainRowsMean[ind], imageMainRowsStd[ind], _, _, _, _ = self.GetImageStats(imageMain)
        #Code to fit a line to the last two points
        #x = np.linspace(len(chInd)-1,len(chInd),len(chInd)+1)
        #p = np.polyfit(x[-3:-1], imageMainRowsMean[-3:-1], 1)
        #imageMainRowsMean[-1] = np.polyval(p, x[-1])
        #p = np.polyfit(x[-3:-1], imageMainRowsStd[-3:-1], 1)
        #imageMainRowsStd[-1] = np.polyval(p, x[-1])
        #self.longScanCorrectionFactor = [imageMainRowsMean, imageMainRowsStd]
        #Code to fit fourth order polynomial to the image mean and std
        x = np.linspace(0,len(chInd)-1,len(chInd))
        pMean = np.polyfit(x, imageMainRowsMean[:-1], 4)
        pStd  = np.polyfit(x, imageMainRowsStd[:-1],  4)
        self.longScanCorrectionFactor = [pMean, pStd]
        return

    def ComputeContrastForChannel(self, channelData, gain, cameraPositionStr, chInd=-1, scanInd=-1):
        scanHistograms, obMeanScan, obStdScan, camTemps, timeStamps = self.ReadHistogram(channelData.histogramPath)
        if len(scanHistograms)==0:
            channelData.dataAvailable = False
            return
        
        imgLaserOff = self.ReadImage(channelData.imagePathLaserOff)
        lsrOffMean, imgMainLsrOffStd, lsrOffWidth, lsrOffObMean, lsrOffObStd, lsrOffObWidth = self.GetImageStats(imgLaserOff)

        imgLaserOn  = self.ReadImage(channelData.imagePathLaserOn)
        lsrOnMean, lsrOnStd, lsrOnWidth, lsrOnObMean, lsrOnObStd, lsrOnObWidth = self.GetImageStats(imgLaserOn)
        
        if self.debugMode > 0:
            # Histogram Stats for Laser On
            channelData.histLsrOnPath          = channelData.histogramPath
            channelData.histMainLsrOn          = scanHistograms
            channelData.histObLsrOnMean        = obMeanScan
            channelData.histObLsrOnStd         = obStdScan
            channelData.histLsrOnCamTemps      = camTemps
            channelData.histLsrOnTimeStamp     = timeStamps
            
            # Image Stats for Laser On
            channelData.imageLaserOffHistWidth = lsrOffWidth
            channelData.imageLaserOffImgMean   = lsrOffMean
            channelData.imgLsrOffPath          = channelData.imagePathLaserOff
            channelData.imgMainLsrOffMean      = lsrOffMean
            channelData.imgMainLsrOffStd       = imgMainLsrOffStd
            channelData.imgMainLsrOffHistWid   = lsrOffWidth
            # channelData.imgMainLsrOffHistRaw = [] # not output currently
            channelData.imgObLsrOffMean        = lsrOffObMean
            channelData.imgObLsrOffStd         = lsrOffObStd
            # channelData.imgObLsrOffHistRaw   = [] # not output currently
            
            # Image Stats for Laser Off
            channelData.imgLsrOnPath           = channelData.imagePathLaserOn
            channelData.imgMainLsrOnMean       = lsrOnMean
            channelData.imgMainLsrOnStd        = lsrOnStd
            channelData.imgMainLsrOnHistWid    = lsrOnWidth
            # channelData.imgMainLsrOnHistRaw  = [] # not output currently
            channelData.imgObLsrOnMean         = lsrOnObMean
            channelData.imgObLsrOnStd          = lsrOnObStd
            # channelData.imgObLsrOnHistRaw    = [] # not output currently
        
        bins = np.array(list(range(scanHistograms.shape[1]-1)))
        scanMean, scanStd, scanHistWidths = self.GetHistogramStats(scanHistograms[:,:-1],bins)
        channelData.histMainLsrOnMean    = scanMean
        channelData.histMainLsrOnStd     = scanStd

        if self.printStats:
            expMeanPr, expStdPr, lsrOffDarkMeanPr, lsrOffDarkStdPr = map(PrettyFloat4, (lsrOffMean, imgMainLsrOffStd, lsrOffObMean, lsrOffObStd))
            print(cameraPositionStr,' Laser off Dark Row Mean:', lsrOffDarkMeanPr, ' Std:', lsrOffDarkStdPr,
                    ' Exposed rows Mean:', expMeanPr, ' Std:', expStdPr )
            print(cameraPositionStr,' Laser on Dark Row Mean:', PrettyFloat4(lsrOnObMean), ' Std:', PrettyFloat4(lsrOnObStd),
                    ' Exposed rows Mean:', PrettyFloat4(lsrOnMean), ' Std:', PrettyFloat4(lsrOnStd),
                    ' Histogram bright mean:', PrettyFloat4(scanMean[0]), ' Std:', PrettyFloat4(scanStd[0]),
                    ' Histogram ob mean:', PrettyFloat4(obMeanScan[0]), ' Std:', PrettyFloat4(obStdScan[0]))

        if self.correctionType==0:
            #mean correction: scan mean - line fit to scan ob rows - offset between main row pixels and ob row pixels in img when laser is off
            #variance correction: scan variance - variance main when laser off - gain*corrected mean(above)
            t = range(0, len(obMeanScan))
            p = np.polyfit(t, obMeanScan, 1)
            obMeanScanFit = np.polyval(p, t)
            channelData.correctedMean = scanMean-(obMeanScanFit+(lsrOffMean-obMeanScanFit[0]))
            if np.any(channelData.correctedMean<0):
                channelData.correctedMean = scanMean-obMeanScanFit
                print('Negative correctedMean in channel. Turning off dark frame offset correction for channel ',cameraPositionStr)
            varCorrected = scanStd**2-imgMainLsrOffStd**2-self.ADCgain*gain*channelData.correctedMean
            if np.any(varCorrected<0):
                varCorrected = scanStd**2
                print('Negative variance in channel with correction. Turning off variance correction for channel ',cameraPositionStr)

        if self.correctionType==1:
            # mean correction - line fit to dark imgMain and dark hist
            # variance corrections from linear fitted dark imgMain and dark hist
            histMainLsrOff, histObLsrOffMean, histObLsrOffStd, histLsrOffCamTemps, histLsrOffTimeStamp = self.ReadHistogram(channelData.darkHistogramPath)
            histMainLsrOffMean, histMainLsrOffStd, histMainLsrOffHistWid = self.GetHistogramStats(histMainLsrOff[:,:-1],bins)
            
            if self.debugMode > 0:
                # Histogram Stats for Laser Off
                channelData.histLsrOffPath        = channelData.darkHistogramPath
                channelData.histMainLsrOff        = histMainLsrOff
                channelData.histObLsrOffMean      = histObLsrOffMean
                channelData.histObLsrOffStd       = histObLsrOffStd
                channelData.histLsrOffCamTemps    = histLsrOffCamTemps
                channelData.histLsrOffTimeStamp   = histLsrOffTimeStamp
                
                channelData.histMainLsrOffMean    = histMainLsrOffMean
                channelData.histMainLsrOffStd     = histMainLsrOffStd
                channelData.histMainLsrOffHistWid = histMainLsrOffHistWid
            
            t = np.array(np.arange(scanHistograms.shape[0]+2))
            polyFit2 = np.poly1d(np.polyfit(np.array([t[0],t[-2],t[-1]]), np.array([lsrOffMean, histMainLsrOffMean[0], histMainLsrOffMean[1],]),1))
            lsrOffMeanFit = polyFit2(t)[:-2]
            polyFit2 = np.poly1d(np.polyfit(np.array([t[0],t[-2],t[-1]]), np.array([imgMainLsrOffStd, histMainLsrOffStd[0], histMainLsrOffStd[1]]),1))
            lsrOffStdFit = polyFit2(t)[:-2]
            channelData.correctedMean  = scanMean-lsrOffMeanFit
            if np.any(channelData.correctedMean<0):
                t = range(0, len(obMeanScan))
                p = np.polyfit(t, obMeanScan, 1)
                obMeanScanFit = np.polyval(p, t)
                channelData.correctedMean = scanMean-obMeanScanFit
                print('Negative correctedMean in channel. Turning off dark frame offset correction for channel ',cameraPositionStr)
            varCorrected = scanStd**2-lsrOffStdFit**2-self.ADCgain*gain*channelData.correctedMean
            if np.any(varCorrected<0):
                varCorrected = scanStd**2-imgMainLsrOffStd**2-self.ADCgain*gain*channelData.correctedMean
                print('Negative variance in channel with correction. Turning off std fit correction for channel ',cameraPositionStr)
                if np.any(varCorrected<0):
                    varCorrected = scanStd**2
                    print('Negative variance in channel with correction. Turning off variance correction for channel ',cameraPositionStr)

        if self.correctionType==2:
            # mean correction - line fit to dark imgMain across all scans for the channel
            # variance corrections - line fit to dark imgMain across all scans for the channel
            if self.longScanCorrectionFactor is None or scanInd==0:
                self.computeCorrectionForLongScansWithNoDarkHists(chInd)
            #Code to fit a line between all the points and to the last two points to extrapolate for the last scan
            #scanMeanCorrection = np.linspace(self.longScanCorrectionFactor[0][scanInd],self.longScanCorrectionFactor[0][scanInd+1],scanMean.shape[0])
            #scanStdCorrection  = np.linspace(self.longScanCorrectionFactor[1][scanInd],self.longScanCorrectionFactor[1][scanInd+1],scanMean.shape[0])
            scanMeanCorrection = np.polyval(self.longScanCorrectionFactor[0], np.linspace(scanInd,scanInd+1,scanMean.shape[0]))
            scanStdCorrection  = np.polyval(self.longScanCorrectionFactor[1], np.linspace(scanInd,scanInd+1,scanMean.shape[0]))
            channelData.correctedMean  = scanMean-scanMeanCorrection
            varCorrected = scanStd**2-scanStdCorrection**2-self.ADCgain*gain*channelData.correctedMean
            if np.any(channelData.correctedMean<0) or np.any(varCorrected<0):
                self.correctionType = 0
                print('Negative correctedMean or variance in channel with correction. Turning off correction for channel ',cameraPositionStr)
                channelData.correctedMean = scanMean
                varCorrected = scanStd**2
        contrast = np.sqrt(varCorrected)/channelData.correctedMean
        channelData.contrastNoFilter = np.copy(contrast)
        if self.filterDrift:
            soshp = signal.butter(2,1/10,'hp',fs=40,output='sos')
            contrastMean = np.mean(contrast)
            contrastHP = signal.sosfiltfilt(soshp, contrast-contrastMean)+contrastMean
            contrast = np.copy(contrastHP)
        if self.filterNoise:
            contrast = denoise_wavelet(contrast, method='BayesShrink', mode='soft', wavelet_levels=6, wavelet='sym3', rescale_sigma='True')
        channelData.contrast = contrast
        channelData.channelPosition = cameraPositionStr
        channelData.lsrOffObWidth = lsrOffObWidth
        channelData.camTemps = camTemps
        channelData.initialCamTemp = camTemps[0]
        channelData.timeStamps = timeStamps

        return

    def ReadDataAndComputeContrast(self):
        self.ReadHistogramAndImageFileNames()

        for ind,channel in enumerate(self.channelData):
            if self.scanType==2:
                indPos = int(ind*2/len(self.channelData))
                scanInd = ind%int(len(self.channelData)/2)
            elif self.scanType==4:
                indPos = int(ind*4/len(self.channelData))
            if self.scanType==2:
                self.ComputeContrastForChannel(channel, self.cameraGain[self.scanType][indPos],
                                               self.cameraPosition[self.scanType][indPos], indPos, scanInd)
            elif self.scanType==4:
                self.ComputeContrastForChannel(channel, self.cameraGain[self.scanType][indPos],
                                               self.cameraPosition[self.scanType][indPos])
            else:
                self.ComputeContrastForChannel(channel, self.cameraGain[self.scanType][ind],
                                               self.cameraPosition[self.scanType][ind])
                
    def WaveformVertNorm(self, x, period):
        # from Soren's containwaveform2 in headscan_gen2_fcns_v5.py
    
        d = np.floor(0.7*period)
        p1 = find_peaks(-x, distance=d)
        p1 = p1[0]
        z1 = x[p1]
        p2 = find_peaks(x, distance=d)
        p2 = p2[0]
        z2 = x[p2]
    
        d = np.floor(0.7*period)
        x = self.flattenbottom(x, d)
        x = self.flattenbottom(x, d)
        x[x<0]=0
        
        #get top peaks
        p = find_peaks(x, distance=d)
        p = p[0]
        if len(p)<2:
            return x
        z = x[p]
        
        g = interp1d(p, z, bounds_error=False, fill_value=(z[0], z[-1]))
        gg = g(np.arange(len(x)))
        gg[:p[0]]=z[0]
        gg[p[-1]:]=z[-1]
        x /= gg
        x[x>1]=1
        
        return x
    
    def flattenbottom(self, x, d):
    
        p = find_peaks(-x, distance=d)
        p = p[0]
        if len(p)<2:
            return x
        z = x[p]
        
        f = interp1d(p, z, bounds_error=False, fill_value=(z[0], z[-1]))
        ff = f(np.arange(len(x)))
        x -= ff
        
        return x
    
    def DetectPulseAndFindFit(self, nofilter=False):
        if nofilter or self.scanType==2 or self.scanType==4: #Relationship between channels is not well defined for these scans
            for loc in range(len(self.channelData)):
                goldenPulseNew, pulseSegmentsNew, onsetsChNew, pulseValidNew, hrNew = self.FindGoldenPulseNew(self.channelData[loc].contrast, filterPulse=False)
                self.channelData[loc].goldenPulse = goldenPulseNew
                self.channelData[loc].pulseSegments = pulseSegmentsNew
                self.channelData[loc].onsets = onsetsChNew
                self.channelData[loc].pulseValid = pulseValidNew
                self.channelData[loc].hr = hrNew
                self.channelData[loc].contrastVertNorm = self.WaveformVertNorm(copy.deepcopy(self.channelData[loc].contrast), np.diff(self.channelData[loc].onsets).mean())
                self.channelData[loc].imgMeanVertNorm = self.WaveformVertNorm(copy.deepcopy(self.channelData[loc].correctedMean), np.diff(self.channelData[loc].onsets).mean())
            return

        if self.scanType==0 or self.scanType==1:
            numScansPos = 4
            pairingIncrement = 4
        elif self.scanType==3: #4 channel scans, left right pairs are 8 apart
            numScansPos = 8
            pairingIncrement = 8
        for loc in range(numScansPos):
            if not self.channelData[loc].dataAvailable or not self.channelData[loc+pairingIncrement].dataAvailable:
                continue
            #Use data from channels where the signal was acquired simultaneously to find the pulse
            y = np.stack((self.channelData[loc].contrast,self.channelData[loc+pairingIncrement].contrast),axis=1)
            N = int(np.power(2,np.ceil(np.log2(y.shape[0]))))  # nearest power of 2 length for FFT
            
            ### approximate period by finding biggest peak in Fourier spectrum ###
            m = np.mean(y, axis=0)
            Y_hat = np.abs(np.fft.fftshift(np.fft.fft(y-m, n=N, axis=0), axes=0))
            mini = np.argmin(Y_hat, axis=0) #Will be freq = 0 because DC is removed before FFT
            maxi = np.argmax(Y_hat, axis=0)
            periodAll = N/np.abs(mini-maxi)
            bpm = 60/self.dt/periodAll
            periodAll[bpm<self.minbpm] = 0
            periodAll[bpm>self.maxbpm] = 0
            period = np.median(periodAll)
            if period == 0 or min(periodAll) == 0:
                print('No waveform at scan location #{} position {}. Skipping bad data'.format(loc+1,self.channelData[loc].channelPosition))
                self.channelData[loc].goldenPulse = np.nan*np.zeros((10,))
                self.channelData[loc+pairingIncrement].goldenPulse = np.nan*np.zeros((10,))
                #pulseDeteced[loc,:] = periodAll != 0
                continue
            else:
                onsets, pulseValid, goldenPulse, pulseSegments = self.FindLowNoisePulseAndAverage(y, period, loc)
                if not goldenPulse or len(goldenPulse)<y.shape[1] or \
                    (len(pulseSegments[0]) and np.nanmedian(np.std(pulseSegments[0],1)) > 0.001) or \
                    (len(pulseSegments[1]) and np.nanmedian(np.std(pulseSegments[1],1)) > 0.001):
                    for ind in range(y.shape[1]):
                        goldenPulseNew, pulseSegmentsNew, onsetsChNew, pulseValidNew, hrNew = self.FindGoldenPulseNew(y[:,ind])
                        if goldenPulseNew is not None and goldenPulse is None:
                            goldenPulse[ind] = goldenPulseNew
                            pulseSegments[ind] = pulseSegmentsNew
                            onsets[ind] = onsetsChNew
                            pulseValid[ind] = pulseValidNew

                        elif goldenPulseNew is not None and goldenPulse is not None and len(goldenPulse)>ind:
                            if not len(pulseSegments[ind]) or \
                                np.nanmedian(np.std(pulseSegmentsNew,1)) < np.nanmedian(np.std(pulseSegments[ind],1)):
                                goldenPulse[ind] = goldenPulseNew
                                pulseSegments[ind] = pulseSegmentsNew
                                onsets[ind] = onsetsChNew
                                pulseValid[ind] = pulseValidNew

                            if len(pulseSegments[ind]) and np.nanmedian(np.std(pulseSegments[ind],1)) > 0.002:
                                goldenPulse[ind] = np.nan*np.zeros((10,))
                                pulseSegments[ind] = []
                                onsets[ind] = []
                                pulseValid[ind] = []

                self.channelData[loc].goldenPulse   = goldenPulse[0]
                self.channelData[loc].pulseSegments = pulseSegments[0]
                self.channelData[loc].onsets   = onsets[0]
                self.channelData[loc].pulseValid    = pulseValid[0]
                self.channelData[loc+pairingIncrement].goldenPulse   = goldenPulse[1]
                self.channelData[loc+pairingIncrement].pulseSegments = pulseSegments[1]
                self.channelData[loc+pairingIncrement].onsets = onsets[1]
                self.channelData[loc+pairingIncrement].pulseValid = pulseValid[1]
                if self.enablePlots:
                    plt.subplot(2, 2, 1)
                    plt.plot(goldenPulse[0])
                    plt.title(self.channelData[loc].channelPosition)
                    plt.subplot(2, 2, 2)
                    plt.plot(y[:,0])
                    plt.title(self.channelData[loc].channelPosition)
                    plt.subplot(2, 2, 3)
                    plt.plot(goldenPulse[1])
                    plt.title(self.channelData[loc+pairingIncrement].channelPosition)
                    plt.subplot(2, 2, 4)
                    plt.plot(y[:,1])
                    plt.title(self.channelData[loc+pairingIncrement].channelPosition)
                    plt.show()

    def FindLowNoisePulseAndAverage(self, y, period, loc):
        '''
            Runs pulse detection and crops, then fits a spline to the pulses in each camera to get the exemplar pulse in each
            camera

            Parameters
            ----------
            y : 2D array
                First dimension is the time the second is the number of channels/cameras

            period : float
                Period(in number of samples) of pulse detected in the signal

            
            Returns
            -------
            fewPulseCount: int
                Count where fewer than three pulses are detected in a channel

            noPulseCount: int
                Count where no pulse is detected in a channel

            fits: list of arrays
                Coefficients for the spline fit
        '''
        noPulseCount  = 0
        fewPulseCount = 0
        ### separate periods ###
        ##Take away all the higher frequency components from the signal to help find the start of systolic
        bpm = 60/self.dt/period
        cut   = np.median(bpm)/60
        soslp = signal.butter(2,cut,'lp',fs=1/self.dt,output='sos')
        m = np.mean(y, axis=0)
        ySmoothed = signal.sosfilt(soslp,y-m,axis=0)
        minPeriod = int(np.floor(period))-int(period*0.15) #allow for a 15% variation decrease in pulse width
        chPeaksSm = [signal.find_peaks(ySmoothed[:,ch], distance=minPeriod)[0] for ch in range(ySmoothed.shape[1])]

        #Make the arrays the same size so that we can use numpy tools on the array
        lenPeaks  = max(map(len,chPeaksSm))
        for tmpInd1, pkList in enumerate(chPeaksSm):
            if len(pkList) == lenPeaks:
                mxLenPeakInd  = tmpInd1
                break
        for tmpInd1 in range(len(chPeaksSm)):
            peaks = chPeaksSm[tmpInd1]
            if len(peaks) < lenPeaks:
                #Append to the correct side - when we have even number of cameras, median can split
                #the difference between the lists and yeild incorrect waveforms
                if np.sum(np.abs(peaks-chPeaksSm[mxLenPeakInd][:len(peaks)])) < \
                    np.sum(np.abs(peaks-chPeaksSm[mxLenPeakInd][-len(peaks):])):
                    for _ in range(lenPeaks-len(peaks)):
                        peaks = np.append(peaks.astype(float),np.nan)
                else:
                    for _ in range(lenPeaks-len(peaks)):
                        peaks = np.insert(peaks.astype(float),0,np.nan)
                chPeaksSm[tmpInd1] = peaks
        
        #Heartbeat should be consistent across cameras
        hbPeaks = np.nanmedian(np.asarray(chPeaksSm),axis=0)
        hbPeaks = hbPeaks.astype(int)
        
        #We search for the systolic peak around the start of the 
        yPrime = np.zeros(y.shape)
        for ind in range(y.shape[1]):
            yPrime[:,ind] = np.gradient(y[:,ind])
        #Primary method to find pulse
        probSystolic1 = [ signal.find_peaks(yPrime[:,yP]**2,distance=minPeriod)[0] for yP in range(yPrime.shape[1]) ]
        #Backup method to find pulse
        probSystolic2 = [ signal.find_peaks(yPrime[:,yP]**2)[0] for yP in range(yPrime.shape[1]) ]
        segStarts = np.zeros((ySmoothed.shape[1],len(hbPeaks)),dtype='i')
        for ind in range(len(hbPeaks)):
            inds1 = probSystolic1 - hbPeaks[ind]
            inds2 = probSystolic2 - hbPeaks[ind]
            for ch,chInds1 in enumerate(inds1):
                chInds1[chInds1>0] = -ySmoothed.shape[0]
                if np.abs(np.amax(chInds1))<5:
                    segStarts[ch,ind] = probSystolic1[ch][np.argmax(chInds1)]
                else:
                    chInds2 = copy.deepcopy(inds2[ch])
                    chInds2[chInds2>0] = -ySmoothed.shape[0]
                    segStarts[ch,ind] = probSystolic2[ch][np.argmax(chInds2)]

        segStarts -=1 #We want the index where the pulse starts
        chPeaks   = [ segSt for segSt in segStarts ]

        chLengths = [peaks[1:]-peaks[:-1] for peaks in chPeaks]

        ### get segments only pick ones that are close < +/-2 of approximation
        chLengthsLst = [l.tolist() for l in chLengths]
        chLengthsLst = sum(chLengthsLst, [])
        maxLength = np.ceil(np.median(chLengthsLst)+1.5)
        minLength = np.floor(np.median(chLengthsLst)-1.5)

        L = int(maxLength+1)

        chUseSegment = [ ( (minLength<=np.array(lengths)).astype(int) + (maxLength>=np.array(lengths)).astype(int) ) == 2
                        for lengths in chLengths ]

        goldenPulse  = [{}] * len(chUseSegment)
        pulseSegments = [{}] * len(chUseSegment)
        pulseDeteced = np.empty((len(chUseSegment)), dtype=bool)
        pulseDeteced.fill(1)

        fits = [[]] * len(chUseSegment)
        for ch in range(len(chUseSegment)):
            nSegments =int(np.sum(chUseSegment[ch]))
            if nSegments == 0:
                goldenPulse[ch] = np.nan*np.zeros((10,))
                pulseDeteced[ch] = False
                noPulseCount += 1
                continue
            segments = np.zeros((L, nSegments))
            if nSegments<3:
                fewPulseCount += 1
            ind=0
            for j in range(len(chLengths[ch])):
                if not chUseSegment[ch][j]:
                    continue
                curSeg = y[chPeaks[ch][j]:chPeaks[ch][j+1]+1, ch]
                if curSeg[0]<curSeg[1]: #We might be off by 1 index from systolic start
                    curSeg = np.roll(curSeg, -1)
                    chPeaks[ch][j] += 1
                segments[:chLengths[ch][j]+1, ind] = curSeg
                ind += 1
            segments[segments == 0] = np.nan
            
            numNan  = np.sum(np.isnan(segments),axis=1)
            delInd0 = np.where(numNan>3)#np.where(numNan>int((segments.shape[1]+0.5)/2))
            #Remove segments that are too long provided there are enough segments to sample
            if delInd0[0].shape[0]>1:
                numNan  = np.sum(~np.isnan(segments[delInd0]),axis=0)
                delInd1 = np.where(numNan>=2)
                segments = np.delete(segments,delInd1,1)
            segments = np.delete(segments,delInd0,0)

            #xArr = np.indices((segments.shape))[0].flatten()
            #segsFlat = segments.flatten()
            #xArr = xArr[~np.isnan(segsFlat)]
            #segsFlat = segsFlat[~np.isnan(segsFlat)]
            #timePoints = np.arange(segments.shape[0])

            if 1:#segments.shape[1]<=2:
                fitOrAvgSegment = np.nanmean(segments-np.nanmean(segments,axis=0),axis=1)+np.nanmean(segments)
            #else:
            #    transformed_x2 = dmatrix("cr(xArr, df=16)",
            #                            {"xArr": xArr}, return_type='matrix')
            #    fit2 = sm.RLM(segsFlat, transformed_x2).fit()
            #    fitOrAvgSegment = fit2.predict(dmatrix("cr(timePoints, df=16)",
            #                            {"timePoints": timePoints}, return_type='matrix'))
            #    fits[ch] = fit2
            goldenPulse[ch] = fitOrAvgSegment
            pulseSegments[ch] = segments-np.nanmean(segments,axis=0)+np.nanmean(segments)
            '''if ch==0:
                plt.plot(y,label=('C1','C2','C3','C4'));plt.title('Location '+str(loc+1)); plt.legend(); plt.show()
            if ch==0 or ch==3:
                plt.plot(segments);plt.title('Location '+str(loc+1)+' Channel '+str(ch));plt.plot(fitOrAvgSegment,color='k'); plt.show()'''
        return chPeaks, chUseSegment, goldenPulse, pulseSegments
    
    def ComputeGoldenPulseFeatures(self):
        for channel in self.channelData:
            channelFeatures = PulseFeatures()
            if channel.goldenPulse is not None:
                channelFeatures.GetWaveformAttributesForSingleChannelPulse(channel.goldenPulse)
            channel.goldenPulseFeatures = channelFeatures
        return
        
    def ComputePulseSegmentFeatures(self):
        for channel in self.channelData:
            if channel.goldenPulse is not None and not np.all(np.isnan(channel.goldenPulse)):
                segmentFeatures = PulseFeatures()
                for segment in channel.pulseSegments.T:
                        segmentFeatures.AppendWaveformAttributesForSingleChannelPulse(segment)
                channel.pulseSegmentsFeatures = segmentFeatures
        return

    def FindPulseLengthNew(self, onsets, printDebug=False):
        lengths = np.convolve(onsets, [1, -1], mode='same')[1:]
        binsLength = np.arange( np.amin(lengths), np.amax(lengths)+1 )
        hist,_ = np.histogram(lengths,bins=binsLength)
        histConv = np.convolve(hist, [1, 1, 1], mode='same')
        indBest  = np.argmax(histConv)
        pulseLength = binsLength[indBest]
        if isinstance(pulseLength, collections.abc.Sequence):
            pulseLength = pulseLength[0]
        
        if printDebug:
            print('Onsets:',onsets)
            print('Pulse lengths:',lengths)
            print('Pulse length histogram:',hist)
            print('Pulse length conv hist:',histConv)
            print('Pulse length hist bins:',binsLength)

            print('Detected pulse length:',pulseLength,'Number of beats:',histConv[indBest])
        return pulseLength, histConv[indBest]

    def GetPulsesNew(self, ch, onsets, pulseLength, numBeatsFound, filterPulse=True):
        #Get pulses with the pulseLength from the input channel ch
        if not filterPulse:
            numBeatsFound = len(onsets)-1
            pulseLength = int(np.max(np.diff(onsets)))
        pulses = np.zeros((numBeatsFound,pulseLength+2))
        ind = 0
        for i in range(1,len(onsets)):
            curLength = onsets[i]-onsets[i-1]
            if filterPulse and (curLength < pulseLength-1.5 or curLength > pulseLength+1.5):
                continue
            pulses[ind,:curLength] = ch[onsets[i-1]:onsets[i]]
            ind += 1
            if ind>=numBeatsFound:
                break
        pulses[np.where(pulses == 0)] = np.nan
        pulseValid = ~np.isnan(pulses).all(axis=0)
        pulses = pulses[:, pulseValid]
        #Waveforms often drift, this removes the drift while averaging and computing features
        if not self.scanType==2 or not self.scanType==4: 
            pulseAvg = np.nanmean(pulses)
            pulses -= np.nanmean(pulses,1)[:,None]
        #Drift correction turned off for use cases where we want to measure changes in the contrast over time
        else:
            pulseAvg = 0
        goldenPulse = np.nanmean(pulses,0)+pulseAvg
        pulses += pulseAvg
        pulses = pulses.T
        return pulses, goldenPulse, pulseValid
    
    def FindGoldenPulseNew(self, ch, showBiosppyPlots=False, filterPulse=True):
        _, filteredCh, onsetsCh, _, hr = biosppy.signals.ppg.ppg(signal=ch, sampling_rate=1/self.dt, show=showBiosppyPlots)
        pulseLengthCh, numBeatsCh = self.FindPulseLengthNew( onsetsCh )
        pulsesCh, goldenPulseCh, pulseValid = self.GetPulsesNew(ch, onsetsCh, pulseLengthCh, numBeatsCh, filterPulse=filterPulse)
        return goldenPulseCh, pulsesCh, onsetsCh, pulseValid, hr

    def PlotContrastMeanAndFrequency(self, titleStr=None, plotContrast=True, plotMean=True, plotGoldenPulses=True, plotFreq=False, plotUnfilteredContrast=False, 
        plotTemp = False):
        if(self.scanType ==0 | self.scanType == 1 ): # Regular scan types with two cameras
            if plotFreq:
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable:
                        continue
                    ft = np.fft.fft(channel.contrast-np.mean(channel.contrast))
                    ft = np.fft.fftshift(ft)
                    freq = np.fft.fftfreq(len(ft),0.025)
                    freq = np.fft.fftshift(freq)
                    plt.subplot(2, 1, 1)
                    plt.plot(freq, np.abs(ft))
                    plt.subplot(2, 1, 2)
                    plt.plot(channel.contrast)
                    plt.suptitle(titleStr)
                    fig = plt.gcf()
                    fig.set_size_inches(18.5, 10.5)
                    plt.show()

            if plotContrast or plotMean:
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable:
                        continue
                    ticks = np.linspace(0,len(channel.contrast)*self.dt,len(channel.contrast))
                    plotInd = (i*2+1)
                    if plotInd>7:
                        plotInd = plotInd - 7
                    ax1 = plt.subplot(4, 2, plotInd)
                    ax2 = ax1.twinx()
                    if plotMean:
                        ax2.plot(ticks,channel.correctedMean,'r')
                        ax2.tick_params(axis='y',color='red')
                        ax2.yaxis.label.set_color('red')
                    if plotContrast:
                        if plotUnfilteredContrast:
                            ax1.plot(ticks,channel.contrastNoFilter,'k')
                        else:
                            ax1.plot(ticks,channel.contrast,'k')
                        ax1.set_zorder(ax2.get_zorder()+1)
                        ax1.set_frame_on(False)
                        ax1.set_xlim(ticks.min(),ticks.max())
                    plt.title(channel.channelPosition)
                plt.suptitle(titleStr)
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)
                plt.show()
                    
            if plotGoldenPulses:
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable or type(channel.pulseSegments) is dict or channel.pulseSegments is None:
                        continue
                    plotInd = (i*2+1)
                    if plotInd>7:
                        plotInd = plotInd - 7
                    ax1 = plt.subplot(4, 2, plotInd)
                    plt.plot(channel.pulseSegments,color='0.8')
                    plt.plot(channel.goldenPulse)
                    plt.title(channel.channelPosition)
                    plt.gca().invert_yaxis()
                    plt.suptitle(titleStr)
                    fig = plt.gcf()
                    fig.set_size_inches(18.5, 10.5)
                plt.show()
            if(plotTemp):
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable:
                        continue
                    ticks = np.linspace(0,len(channel.camTemps)*self.dt,len(channel.camTemps))
                    plotInd = (i*2+1)
                    if plotInd>7:
                        plotInd = plotInd - 7
                    ax1 = plt.subplot(4, 2, plotInd)
                    ax1.plot(ticks,channel.camTemps,'g')
                    ax1.tick_params(axis='y',colors='green')
                    ax1.yaxis.label.set_color('green')
                    ax1.set_xlim(ticks.min(),ticks.max())
                    plt.title(channel.channelPosition)
                plt.suptitle(titleStr)
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)
                plt.show()
        elif(self.scanType == 2 or self.scanType == 4): # Long scan types
            self.PlotLongScan(titleStr=titleStr, plotContrast=plotContrast, plotMean=plotMean)
        elif(self.scanType == 3): # 4 channel simultaneous scan type
            if plotContrast or plotMean:
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable:
                        continue
                    ticks = np.linspace(0,len(channel.contrast)*self.dt,len(channel.contrast))
                    # plotInd = (i*2+1)
                    plotInd = i + 1
                    ax1 = plt.subplot(4, 4, plotInd)
                    ax2 = ax1.twinx()
                    if plotMean:
                        ax2.plot(ticks,channel.correctedMean,'r')
                        ax2.tick_params(axis='y',colors='r')
                        ax2.yaxis.label.set_color('r')
                    if plotContrast:
                        if plotUnfilteredContrast:
                            ax1.plot(ticks,channel.contrastNoFilter,'k')
                        else:
                            ax1.plot(ticks,channel.contrast,'k')
                        ax1.set_zorder(ax2.get_zorder()+1)
                        ax1.set_frame_on(False)
                        ax1.set_xlim(ticks.min(),ticks.max())
                    plt.title(channel.channelPosition)
                plt.suptitle(titleStr)
                fig = plt.gcf()
                fig.set_size_inches(37.0, 10.5)
                plt.show()
            fig.savefig('/Users/brad/Desktop/figs/' + titleStr.split('/')[-1] + '.png',dpi=300)
                    
            if plotGoldenPulses:
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable or type(channel.pulseSegments) is dict or channel.pulseSegments is None:
                        continue
                    plotInd = (i*2+1)
                    if plotInd>7:
                        plotInd = plotInd - 7
                    ax1 = plt.subplot(4, 2, plotInd)
                    plt.plot(channel.pulseSegments,color='0.8')
                    plt.plot(channel.goldenPulse)
                    plt.title(channel.channelPosition)
                    plt.gca().invert_yaxis()
                    plt.suptitle(titleStr)
                    fig = plt.gcf()
                    fig.set_size_inches(18.5, 10.5)
                plt.show()
            if(plotTemp):
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable:
                        continue
                    ticks = np.linspace(0,len(channel.camTemps)*self.dt,len(channel.camTemps))
                    plotInd = (i+1)
                    ax1 = plt.subplot(4, 4, plotInd)
                    ax2 = ax1.twinx()
                    ax1.plot(ticks,channel.camTemps,'g')
                    ax1.tick_params(axis='y',colors='green')
                    ax1.yaxis.label.set_color('green')
                    plt.title(channel.channelPosition)
                plt.suptitle(titleStr)
                fig = plt.gcf()
                fig.set_size_inches(18.5, 10.5)
                plt.show()
    
    def PlotInvertedAndCenteredContrastAndMean(self, titleStr=None, plotContrast=True, plotMean=True, plotUnfilteredContrast=False):
        for i,channel in enumerate(self.channelData):
            if not channel.dataAvailable:
                continue
            ticks = np.linspace(0,len(channel.contrast)*self.dt,len(channel.contrast))
            if self.scanType<2:
                plotInd = (i*2+1)
                if plotInd>7:
                    plotInd = plotInd - 7
                ax1 = plt.subplot(4, 2, plotInd)
            elif self.scanType == 3:
                plotInd = (i+1)
                ax1 = plt.subplot(4, 4, plotInd)
            ax2 = ax1.twinx()
            if plotMean:
                ax2.plot(ticks,channel.correctedMean - np.mean(channel.correctedMean),'r')
                ax2.tick_params(axis='y',colors='red')
                ax2.yaxis.label.set_color('red')
                ax2.invert_yaxis()
            if plotContrast:
                if plotUnfilteredContrast:
                    ax1.plot(ticks,channel.contrastNoFilter - np.mean(channel.contrastNoFilter),'k')
                else:
                    ax1.plot(ticks,channel.contrast - np.mean(channel.contrast),'k')
                ax1.invert_yaxis()
                ax1.set_zorder(ax2.get_zorder()+1)
                ax1.set_frame_on(False)
                ax1.set_xlim(ticks.min(),ticks.max())
            plt.title(channel.channelPosition)
        plt.suptitle(titleStr)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()

    def PlotLongScan(self, titleStr=None, saveFig=[], cropTime=[]):
        numScans = len(self.channelData)
        if self.scanType==2:
            numScansPerChannel = int(np.ceil(numScans/2))
            plotOrder = [1,2]
            numChannels = 2
        elif self.scanType==4:
            numScansPerChannel = int(np.ceil(numScans/4))
            plotOrder = [1,4,2,3]
            numChannels = 4
        for i in range(numChannels):
            plotScanList = []
            for j in range(numScansPerChannel):
                ch = self.channelData[i*numScansPerChannel+j]
                if not ch.dataAvailable:
                    continue
                plotScanList.append(ch)
            if len(plotScanList)==0:
                continue
            ax1 = plt.subplot(numChannels, 1, plotOrder[i])
            ax2 = ax1.twinx()
            contBaseline = plotScanList[0].contrast.mean()
            meanBaseline = plotScanList[0].correctedMean.mean()
            for j,channel in enumerate(plotScanList):
                ticks = np.linspace(0,len(channel.contrast)*self.dt,len(channel.contrast))
                timeSet = True
                if 1:#channel.timeStamps is None:
                    ticks += j*(ticks[-1]+0.714) #Estimating 3 seconds between scans
                    timeSet = False
                elif len(channel.timeStamps)==1:
                    ticks += channel.timeStamps
                else:
                    ticks = channel.timeStamps
                if timeSet:
                    #Plot with timestamps in minutes and seconds
                    contFilt = copy.deepcopy(channel.contrast)
                    ax1.plot(ticks,contBaseline-contFilt,'k')
                    
                    #Plot with timestamps in minutes, seconds, and milliseconds
                    dates = [dt.datetime.fromtimestamp(ts/1000000000) for ts in ticks]
                    datenums=md.date2num(dates)
                    xfmt = md.DateFormatter('%M:%S.%f')
                    #set x axis ticks 200 observations apart
                    ax2.xaxis.set_major_formatter(xfmt)
                    ax2.xaxis.set_major_locator(md.MinuteLocator(interval=1))
                    ax2.plot(datenums,meanBaseline-channel.correctedMean,'r')
                    ax2.tick_params(axis='y',colors='r')
                else:
                    contFilt = copy.deepcopy(channel.contrast)
                    ax1.plot(ticks,contBaseline-contFilt,'k')
                    
                    ax2.plot(ticks,meanBaseline-channel.correctedMean,'r')
                    ax2.tick_params(axis='y',colors='r')
                ax1.set_ylabel('rBFI')
                ax2.set_ylabel('rBVI',color='r')
                if cropTime != []:
                    ax1.set_xlim(cropTime)
                    ax2.set_xlim(cropTime)
                if plotMean:
                    if timeSet:
                        #Plot with timestamps in minutes, seconds, and milliseconds
                        dates = [dt.datetime.fromtimestamp(ts/1000000000) for ts in ticks]
                        datenums=md.date2num(dates)
                        xfmt = md.DateFormatter('%M:%S.%f')
                        #set x axis ticks 200 observations apart
                        ax2.xaxis.set_major_formatter(xfmt)
                        ax2.xaxis.set_major_locator(md.MinuteLocator(interval=1))
                        ax2.plot(datenums,channel.correctedMean,'r')
                    else:
                        ax2.plot(ticks,channel.correctedMean,'r')
                        ax2.tick_params(axis='y',colors='red')
                        ax2.yaxis.label.set_color('red')
                if plotContrast:
                    if timeSet:
                        #Plot with timestamps in minutes and seconds
                        ax1.plot(ticks,channel.contrast,'k')
                    else:
                        ax1.plot(ticks,channel.contrast,'k')
                    ax1.set_zorder(ax2.get_zorder()+1)
                    ax1.set_frame_on(False)
                plt.title(channel.channelPosition)
        plt.suptitle(titleStr)
        fig = plt.gcf()
        fig.set_size_inches(12, 9)
        if saveFig != []:
                fig.savefig(saveFig + titleStr + '_plotContMean.png',dpi=300,bbox_inches='tight')
        plt.show()
        
    def PlotCompare4Channels(self, titleStr=None, saveFig=[], plotGoldenPulses=False):
        if(self.scanType ==0 | self.scanType == 1 ): # Regular scan types with two cameras
            print('No')
        elif(self.scanType == 2 or self.scanType == 4): # Long scan types
            print('Also No')
        elif(self.scanType == 3): # 4 channel simultaneous scan type
            trashNames = ['RH','RH','LN','RH',
                        'LN','RV','RV','LN',
                        'LH','LH','RN','LH',
                        'RN','LV','LV','RN',] #4 channel scan
            trashNames = np.array(trashNames)
            linColor = ['b','b','r','b',
                        'r','r','b','r',
                        'b','b','r','b',
                        'r','r','b','r',] #4 channel scan (red always near)
            linThick = [2,2,1,2,
                        1,2,2,1,
                        1,1,2,1,
                        2,1,1,2,] #4 channel scan (left always thin)
            linThick = np.array(linThick)/2
            
            for i,channel in enumerate(self.channelData):
                if not channel.dataAvailable:
                    continue
                ticks = np.linspace(0,len(channel.contrast)*self.dt,len(channel.contrast))
                plotInd = i%4*2 + 1
                ax1 = plt.subplot(4, 2, plotInd)
                if plotGoldenPulses:
                    # dataToPlot = channel.pulseSegments
                    dataToPlot = channel.goldenPulse
                    if not channel.dataAvailable or type(channel.pulseSegments) is dict or channel.pulseSegments is None:
                        continue
                    ax1.plot(dataToPlot,linColor[i],linewidth=linThick[i])
                else:
                    dataToPlot = channel.contrastVertNorm # contrastVertNorm contrast contrastNoFilter 
                    dataToPlot = dataToPlot.mean() - dataToPlot
                    dataToPlot = dataToPlot - dataToPlot.min()
                    dataToPlot = dataToPlot / dataToPlot.max()
                    ax1.plot(ticks,dataToPlot,linColor[i],linewidth=linThick[i])
                
                ax1 = plt.subplot(4, 2, plotInd + 1)
                dataToPlot = channel.imgMeanVertNorm # imgMeanVertNorm correctedMean 
                dataToPlot = dataToPlot.mean() - dataToPlot
                dataToPlot = dataToPlot - dataToPlot.min()
                dataToPlot = dataToPlot / dataToPlot.max()
                ax1.plot(ticks,dataToPlot,linColor[i],linewidth=linThick[i])
                
            for rowInd in range(4):
                ax1 = plt.subplot(4, 2, rowInd*2+1)
                ax1.legend(trashNames[np.array([0,4,8,12])+rowInd],loc='upper right')
                ax1 = plt.subplot(4, 2, rowInd*2+2)
                ax1.legend(trashNames[np.array([0,4,8,12])+rowInd],loc='upper right')
            
            plt.subplot(4, 2, 1).set_title('Contrast')
            plt.subplot(4, 2, 2).set_title('Image Mean')
            
            plt.suptitle(titleStr)
            fig = plt.gcf()
            fig.set_size_inches(37.0, 10.5)
            if saveFig != []:
                fig.savefig(saveFig + titleStr + '_compare4Ch.png',dpi=300,bbox_inches='tight')
            plt.show()
                
            if plotGoldenPulses:
                for i,channel in enumerate(self.channelData):
                    if not channel.dataAvailable or type(channel.pulseSegments) is dict or channel.pulseSegments is None:
                        continue
                    # plotInd = (i*2+1)
                    # if plotInd>7:
                    #     plotInd = plotInd - 7
                    # ax1 = plt.subplot(4, 2, plotInd)
                    # plt.plot(channel.pulseSegments,color='0.8')
                    # plt.plot(channel.goldenPulse)
                    # plt.title(channel.channelPosition)
                    # plt.gca().invert_yaxis()
                    # plt.suptitle(titleStr)
                    # fig = plt.gcf()
                    # fig.set_size_inches(18.5, 10.5)
                print('No support for Golden Pulses')
                plt.show()
                
    def PlotLongScanCompare4Channels(self, titleStr=None, saveFig=[], cropTime=[]):
        numScans = len(self.channelData)
        if self.scanType==2:
            # numScansPerChannel = int(np.ceil(numScans/2))
            # plotOrder = [1,2]
            # numChannels = 2
            return
        elif self.scanType==4:
            numScansPerChannel = int(np.ceil(numScans/4))
            plotOrder = [1,4,2,3]
            numChannels = 4
            
        trashNames = ['RH','LN','LH','RN',] #4 channel scan
        linColor = ['b','r','b','r',] #4 channel scan (red always near)
        linThick = [2,1,1,2,] #4 channel scan (left always thin)
        linThick = np.array(linThick)/2
        
        for i in range(numChannels):
            plotScanList = []
            for j in range(numScansPerChannel):
                ch = self.channelData[i*numScansPerChannel+j]
                if not ch.dataAvailable:
                    continue
                plotScanList.append(ch)
            if len(plotScanList)==0:
                continue
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
            
            for j,channel in enumerate(plotScanList):
                ticks = np.linspace(0,len(channel.contrast)*self.dt,len(channel.contrast))
                timeSet = True
                if 1:#channel.timeStamps is None:
                    ticks += j*(ticks[-1]+0.714) #Estimating 3 seconds between scans
                    timeSet = False
                elif len(channel.timeStamps)==1:
                    ticks += channel.timeStamps
                else:
                    ticks = channel.timeStamps

                if timeSet:
                    #Plot with timestamps in minutes and seconds
                    ax1.plot(ticks,1-channel.contrastVertNorm,linColor[i],linewidth=linThick[i])
                    
                    #Plot with timestamps in minutes, seconds, and milliseconds
                    dates = [dt.datetime.fromtimestamp(ts/1000000000) for ts in ticks]
                    datenums=md.date2num(dates)
                    xfmt = md.DateFormatter('%M:%S.%f')
                    #set x axis ticks 200 observations apart
                    ax2.xaxis.set_major_formatter(xfmt)
                    ax2.xaxis.set_major_locator(md.MinuteLocator(interval=1))
                    ax2.plot(ticks,1-channel.imgMeanVertNorm,linColor[i],linewidth=linThick[i])
                    
                else:
                    ax1.plot(ticks,1-channel.contrastVertNorm,linColor[i],linewidth=linThick[i])
                    ax2.plot(ticks,1-channel.imgMeanVertNorm,linColor[i],linewidth=linThick[i])
                
                ax1.set_ylabel('Normalized Contrast')
                ax2.set_ylabel('Normalized Image mean')    
                custom_lines = [Line2D([0], [0], color='b', lw=2),
                                Line2D([0], [0], color='r', lw=1),
                                Line2D([0], [0], color='b', lw=1),
                                Line2D([0], [0], color='r', lw=2),]
                ax1.legend(custom_lines, trashNames, loc='lower right')
                ax2.legend(custom_lines, trashNames, loc='lower right')
                if cropTime != []:
                    ax1.set_xlim(cropTime)
                    ax2.set_xlim(cropTime)
                ax2.set_xlabel('Time (s)')
        plt.suptitle(titleStr)
        fig = plt.gcf()
        fig.set_size_inches(24, 9)
        if saveFig != []:
            fig.savefig(saveFig + titleStr + '_compare4Ch.png',dpi=300,bbox_inches='tight')
        plt.show()
        
    def displayRawImage(self,chInd,lsrOnOff=1,saveDir=''):
        # channel index, laser on or off, save directory (if exists will save) as input
        if lsrOnOff == 1:
            img = self.ReadImage(self.channelData[chInd].imagePathLaserOn)
            lsrOnOffStr = 'On'
        else:
            img = self.ReadImage(self.channelData[chInd].imagePathLaserOff)
            lsrOnOffStr = 'Off'
        numCam = 4
        numMod = int(len(self.channelData)/numCam)
        
        fig, ax = plt.subplots(layout='constrained')
        fig.tight_layout()
        # color bar limits set to +/- 1 S.D. from mean
        ax1 = ax.imshow(img, clim=(img.mean()-img.std(),img.mean()+img.std()))
        ax.set_title(self.path.split('/')[-2] + '/' + self.path.split('/')[-1] + 
                  '\n Camera Channel ' + str(chInd) + ' (Mod' + str(chInd%numMod) + ' Cam' + str(chInd//numMod) +
                  ' ' + self.channelData[chInd].channelPosition + '), Laser ' + lsrOnOffStr)
        if lsrOnOff == 1:
            ax.set_xlabel('Image Mean (main,OB): ' + str(self.channelData[chInd].imgMainLsrOnMean.round(2)) + 
                   ', ' + str(self.channelData[chInd].imgObLsrOnMean.round(2)) + 
                   '\nImage S.D. (main,OB): ' + str(self.channelData[chInd].imgMainLsrOnStd.round(2)) + 
                   ', ' + str(self.channelData[chInd].imgObLsrOnStd.round(2)) + 
                   '\nHot/bad pixels displayed but removed for stats (<' + str(self.noisyBinMin) + ' ct)')
        else:
            ax.set_xlabel('Image Mean (main,OB): ' + str(self.channelData[chInd].imgMainLsrOffMean.round(2)) + 
                   ', ' + str(self.channelData[chInd].imgObLsrOffMean.round(2)) + 
                   '\nImage S.D. (main,OB): ' + str(self.channelData[chInd].imgMainLsrOffStd.round(2)) + 
                   ', ' + str(self.channelData[chInd].imgObLsrOffStd.round(2)) + 
                   '\nHot/bad pixels displayed but removed for stats (<' + str(self.noisyBinMin) + ' ct)')
        fig.colorbar(ax1,shrink=0.5)
        if saveDir != '':
            fig.savefig(saveDir + 'img_' + 'Ch' + str(chInd) + 'Mod' + str(chInd%numMod) + 'Cam' + str(chInd//numMod)
                        + 'Lsr' + lsrOnOffStr + '.png', dpi=300,bbox_inches='tight')
        
        return
