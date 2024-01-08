# This function primarily loads up tediusly long lists

def LongScan_breathHoldPaper():
    
    shortNames = [
            'Test01','Test02','Test03','CO2_01','CO2_02',
            'Subj01','Subj02','Subj03','Subj04','Subj05',
            'Subj06','Subj07','Subj08','Subj09',
            'Subj10','Subj12','Subj13','Subj14','Subj15',
            'Subj16','Subj17','Subj18','Subj19',
            'Subj20','Subj21','Subj22','Subj23',
            ]
    
    scanNames = [
        '2023_02_06_094608_CVRtest01/LONGSCAN_2023_02_06_095646', # modified to be LONG from TEST
        # '2023_02_06_101001_Phantom/TESTSCAN_2023_02_06_101556',
        '2023_02_06_110259_CVRbh_test2/LONGSCAN_2023_02_06_111200', # modified to be LONG from TEST
        '2023_02_06_114904_CVRbh_test3/LONGSCAN_2023_02_06_115526', # modified to be LONG from TEST
        # '2023_02_06_125040_Phantom/TESTSCAN_2023_02_06_125302',
        '2023_02_07_140642_CVRco2_test/LONGSCAN_2023_02_07_141248', # modified to be LONG from TEST
        '2023_02_07_140642_CVRco2_test/LONGSCAN_2023_02_07_142511', # modified to be LONG from TEST
        
        '2023_03_02_103948_CVRhBH01/LONGSCAN_2023_03_02_105508', # modified to be LONG from TEST
        '2023_03_02_110335_CVRhBH02/LONGSCAN_2023_03_02_111206', # modified to be LONG from TEST
        '2023_03_02_112510_CVRhBH03/LONGSCAN_2023_03_02_113454', # modified to be LONG from TEST
        '2023_03_02_114424_CVRhBH04/LONGSCAN_2023_03_02_115656', # modified to be LONG from TEST
        '2023_03_02_120331_CVRhBH05/LONGSCAN_2023_03_02_121230', # modified to be LONG from TEST
        # # '2023_03_02_121828_Phantom/FULLSCAN_2023_03_02_121914',
        
        '2023_04_12_110319_CVRhBH06/LONGSCAN_2023_04_12_114340',
        '2023_04_12_121006_CVRhBH07/LONGSCAN_2023_04_12_122259',
        # '2023_04_12_123054_Phantom/FULLSCAN_2023_04_12_123129',
        '2023_04_13_143327_CVRhBH08/LONGSCAN_2023_04_13_145315',
        '2023_04_13_150053_CVRhBH09/LONGSCAN_2023_04_13_151249',
        
        #'2023_05_08_122354_Phant_new_confg/FULLSCAN_4C_2023_05_08_122437',
        '2023_05_08_132901_CVRhBH010_2/LONGSCAN_4C_2023_05_08_133407',
        '2023_05_08_143239_CVRhBH012/LONGSCAN_4C_2023_05_08_145232',
        '2023_05_09_131610_CVRhBH013/LONGSCAN_4C_2023_05_09_133305',
        '2023_05_09_134313_CVRhBH014/LONGSCAN_4C_2023_05_09_135335',
        '2023_05_09_140841_CVRhBH015/LONGSCAN_4C_2023_05_09_142301',
        
        '2023_05_25_135423_CVRhBH016/LONGSCAN_4C_2023_05_25_141857',
        '2023_05_25_142850_CVRhBH017/LONGSCAN_4C_2023_05_25_144228',
        # '2023_05_25_145253_Phantom/'FULLSCAN_4C_2023_05_25_145345,
        '2023_05_25_150054_CVRhBH018/LONGSCAN_4C_2023_05_25_151111',
        '2023_05_25_153003_CVRhBH019/LONGSCAN_4C_2023_05_25_154016',
        
        '2023_06_07_143015_CVRhBH020/LONGSCAN_4C_2023_06_07_145039',
        '2023_06_07_150441_CVRhBH021/LONGSCAN_4C_2023_06_07_151408',
        '2023_06_07_152515_CVRhBH022/LONGSCAN_4C_2023_06_07_153250',
        '2023_06_07_153824_CVRhBH023/LONGSCAN_4C_2023_06_07_155057',
        # '2023_06_07_155816_Phantom/FULLSCAN_4C_2023_06_07_155829',
         ]
    
    bilatTypes = [2] # Type of bilateral or not scanning setup used
    
    varCorTypes = [2] # Type of variance correction to be used
    
    lsTypes = [
        1,1,1,1,1, # Long scan type used (needed for various processing types)
        1,1,1,1,1,
        2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,
        3,3,3,3,
        ]
    
    tcdNames = [
        'CVR long scan Breath hold 01.txt', #0
        'CVR 2 breath hold TCD file.txt',
        'CVR Breath hold test 3 (TCD).txt',
        'CVR CO2 TEST 01.txt',
        'CVR CO2 test 02.txt',
        
        'CVRhBH01.txt', #5
        'CVRhBH02.txt',
        'CVRhBH03.txt',
        'CVRhBH04.txt',
        'CVRhBH05.txt',

        'CVRhBH06_updated.txt', #10
        'CVRhBH07_updated.txt',
        'CVRhBH08_updated.txt',
        'CVRhBH09_updated.txt',

        'CVRhBH10.txt', #14
        'CVRhBH12.txt',
        'CVRhBH13.txt',
        'CVRhBH14.txt',
        'CVRhBH15.txt',
        
        'CVRhBH16.txt', #19
        'CVRhBH17.txt',
        'CVRhBH18.txt',
        'CVRhBH19.txt',
        
        'CVRhBH020.txt', #23
        'CVRhBH021.txt',
        'CVRhBH022.txt',
        'CVRhBH023.txt',
        ]
    
    # Values from TCD text file indicating approximate start/end of hold
    tcdMarks = [ # [start optical, hold, breathe, hold, breathe, end minus extras beyond optical scan]
        [-3300,15192,18426,28491-0], #ignore 10378 #hold 15192 # breathe 18426 28491
        [13623,22112,26063,39343-3400],
        [6236,13490,17860,28074-2200],
        [4039,12799,26410,38859-4400],
        [7628,15640,29036,43457-6579], #-5700
        
        [11099,19630,23730,31189,35272,40668-319], # baseline, hold, breathe, hold-FAIL, breathe, stop
        [-500,7985,11711,19362,23757,28794-0], # OW began -4s, hold, breathe, hold, breathe, stop
        [71696+35,80110,84778,93157,97345,101419-473], # start, hold, breathe, hold, breathe, stop
        [96213,104006,107956,116219,120209,126489-1026], # begin, hold, breathe, hold, breathe, stop
        [9192,16249,19175-625,29731,33109,39303-861], # start, hold, breathe, hold, breathe, Mark
        # -625 used to truncate loss of signal after breathing
        
        [0,7633,11919-697,21132,24902,31815], # baseline, bh, breath, bh, breath, stop
        # -697 used to truncate noisey/loss of signal after breathing
        [0,5731,9532,21452,25237,29627], # baseline, hold, breathe, hold, breathe, stop
        [0,16748,20697,25324,27522,29754], # start scan, hold, breathe, hold, breathe, stop scan
        [0,7887,10069,20523,22226,29781], # begin scan, jiggle(1595), hold, breath, hold, breath, stop scan
        
        [0,8303,12131,20444,23999,27583], # begin long scan,hold,breathe,hold,breathe,end
        [0,9260,13063,21596,25507,29750], # begin long scan,hold,breath,hold,breath,end scan
        [0,10819,14277,21484,25594,29781], # BEGIN,HOLD,BREATHE,HOLD,BREATHE,END SCAN
        [0,8026,11904,20268,24231,30366], # BEGIN,HOLD,BREATHE,HOLD,BREATHE,END SCAN
        [0,8286,11293,18733,21941,29780], # BEGIN,HOLD,BREATHE,HOLD,BREATHE,END SCAN
        
        [0,8240,10994,20030,23264-401,29713], # begin long scan,hold,breath,hold,breath,end long scan
        # -401 used to truncate noisey/loss of signal after breathing
        [0,8014,9524,17011,19070,23869,25980,29680], # long scan,co2(4374),hold,breath,hold,breath,hold,breath,end long scan
        [0,8280,11956,19632,23330,29710], # long scan,hold,breath,hold,breath,end long scan
        [0,8866,10935,19212,21075,29788], # begin long scan,hold,breath,hold,breath,jiggle(25681),end long scan
        
        [0,9567,12019,21097,23199,29756], # begin long scan,ad etco2(4086),hold,breath,hold,breath,end long scan
        [0,9684,13110,22005,25757,29735], # strt long scan,hold,breath,hold,breath,end long scan
        [0,8774,12682,22573,26449,29770], # strt long scan,hold,breath,hold,breath,end long scan
        [0,8869,12156,22419,25113,29858], # start long scan,hold,breath,spoke(14745),hold,breath,end long scan
        ]
    
    # Offsets generated from auto-aligning HR signals from optical and TCD
    tcdMarksOffset = [
        [-119], #[0,0],
        [+238], #[-256,0], #4
        [+237], #[-253,0],
        [-44], #[-16,0],
        [-91], #[+59,0],
        
        [+233], #[-250,-272], #4
        [+13 ], #[0,0],       #4
        [+171], #[-222,-80],  #4
        [+110], #[-131,-131], #4 raw [-131, -625]
        [+204], #[-222,-222], #4 raw [-222, -625] # used to be +641
        
        [+214], #[-247,-247], #2 raw [-466, -247]
        [+227], #[-238,+53],  #4 raw [-238, +53]
        [+225], #[-253,-262], #4
        [+214], #[-228,-247], #4
        
        [+231], # manually determined, lowest value not used
        [+222], # 304
        [+233], # 
        [+215], # 
        [+238], # -302
        
        [+222], # 
        [+216], # 
        [+223], # 
        [+230], # 244 aligns pusles: 335 (TCD one pulse to left of optical at jiggle), 230 for best alignment at jiggle
        
        [+228], # 128
        [+238], #
        [+157], #
        [+236], #
        ]
    
    # Manual peak locations used (left as 0 if auto-detected) for averaged waveform analysis
    peakLocs = [ # rBFI-pre rBFI-post TCD-pre TCD-post
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Test_1
                [[0    ,0    ,0    , 0.300,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Test_2
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Test_3
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # CO2_1
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # CO2_2
                
                [[0.290,0    ,0    , 0    ,0    ,0    ],[0.270,0    ,0    , 0.290,0    ,0.540]], # Sub_01
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_02
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_03
                [[0    ,0    ,0    , 0.290,0    ,0    ],[0    ,0    ,0    , 0    ,0.400,0    ]], # Sub_04
                [[0    ,0.400,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_05
                
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_06
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_07
                [[0    ,0    ,0    , 0    ,0.430,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_08
                [[0    ,0    ,0    , 0.280,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_09
                
                [[0    ,0.420,0    , 0    ,0.400,0    ],[0    ,0.000,0    , 0    ,0    ,0    ]], # Sub_10
                [[0    ,0    ,0.540, 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_12
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_13
                [[0    ,0    ,0    , 0    ,0    ,0.570],[0    ,0    ,0    , 0    ,0    ,0.610]], # Sub_14
                [[0    ,0    ,0    , 0    ,0.000,0.000],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_15
                
                [[0    ,0.380,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0.000,0.540]], # Sub_16
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_17
                [[0.000,0    ,0    , 0.300,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_18
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_19
                
                [[0    ,0.395,0    , 0    ,0    ,0    ],[0    ,0.000,0    , 0    ,0    ,0    ]], # Sub_20
                [[0    ,0.425,0    , 0    ,0.420,0    ],[0    ,0.420,0    , 0    ,0.400,0    ]], # Sub_21
                [[0    ,0    ,0    , 0    ,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_22
                [[0    ,0    ,0    , 0.290,0    ,0    ],[0    ,0    ,0    , 0    ,0    ,0    ]], # Sub_23
                ]
    
    # Manual trough locations used (left as 0 if auto-detected) for averaged waveform analysis
    trofLocs = [ # rBFI-pre rBFI-post TCD-pre TCD-post
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Test_1
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Test_2
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Test_3
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # CO2_1
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # CO2_2
                
                [[0    ,0    , 0.315,0    ],[0.315,0    , 0.320,0.510]], # Sub_01
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0.370,0    ]], # Sub_02
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_03
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0.350,0    ]], # Sub_04
                [[0.360,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_05
                
                [[0    ,0    , 0    ,0    ],[0.000,0    , 0    ,0    ]], # Sub_06
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_07
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_08
                [[0    ,0    , 0.325,0    ],[0    ,0    , 0    ,0    ]], # Sub_09
                
                [[0.360,0    , 0.345,0    ],[0.350,0    , 0    ,0    ]], # Sub_10
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_12
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_13
                [[0    ,0.533, 0    ,0    ],[0    ,0    , 0    ,0.540]], # Sub_14
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_15
                
                [[0.345,0    , 0    ,0    ],[0.360,0    , 0.000,0.000]], # Sub_16
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_17
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_18
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_19
                
                [[0.350,0    , 0    ,0    ],[0.000,0    , 0    ,0    ]], # Sub_20
                [[0.380,0    , 0.380,0    ],[0.380,0    , 0.360,0    ]], # Sub_21
                [[0    ,0    , 0    ,0    ],[0    ,0    , 0    ,0    ]], # Sub_22
                [[0    ,0    , 0.310,0    ],[0    ,0    , 0    ,0    ]], # Sub_23
                ]
    
    return shortNames,scanNames,bilatTypes,varCorTypes,lsTypes,tcdNames,tcdMarks,tcdMarksOffset,peakLocs,trofLocs