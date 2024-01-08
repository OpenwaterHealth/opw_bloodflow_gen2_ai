Overview of files used:

headscan_gen2_paper.py
Run this file to generate allData_contMeanPulses.pkl and generate example data plots and waveform analysis
Set processCont and processPulses to True the first time running, after that can be set to False to save time
Update data/saving directories as appropriate

processAllTcdPaper.py
Run this file to generate all correlation plots between optical and TCD datasets
This file relies on allData_contMeanPulses.pkl, so headscan_gen2_paper.py must be run first
Update data/saving directories as appropriate

batchfilePaper.py
Ancillary function to load large lists of variables

headscan_gen2_fcns_paper.py
Wide variety of functions to process and plot the data



Environment requirements:
Anaconda version  4.13.0
Spyder version 4.2.5