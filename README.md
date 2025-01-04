## SigAnTo

A lightweight signal analysis toolbox GUI.<br><br>
Built on top of base python (>=3.9 with tkinter), numpy, matplotlib and scipy.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university or industry environments (or use the executable to avoid dependencies altogether).

Its purpose is to read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs and measurements.
Real-time applications are outside of the scope of this tool.<br>
Some of the identification is automated (see examples further down) but should always be confirmed manually.<br>

The tool only supports analysis for now, fine-tuning and demodulation will be covered in further versions.<br>
In these, I might also switch from Tkinter to PyQt for better performance ; be aware that you will likely experience some sluggishness if your file contains samples in the order of a few millions and the GUI will become downright painful to use with a file containing several tens of millions of samples or more.<br>

Expected file encoding is standard 16-bit wav, but 8 to 64-bit is supported. Other formats might be added if there is a need for it.<br>

The repository also contains 2 scripts to convert mp3 and SigMF files into wav, so it can then be read into SigAnTo (or whatever else shares .wav requirements). Be aware the mp3 conversion requires ffmpeg.<br>

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/><br>

### Summary of available functions <br>
#### 1. Display
- Load a .wav file. The button reloads the same file if clicked again. Select the button next to it to close the .wav file before loading another.
- Activate measuring cursors (deactivated by default upon loading a new graph). Distance between 2 cursors is shown and you can also jump to a nearby peak.
- Change FFT window size (default 512)
- Change language (english and french available)
- Display frequency & power information of the signal file (Estimated BW, dB level, symbol rate & ACF)
- Modify parameters for transition (phase, frequency) & persistence graphs
#### 2. Signal modification
- Low, High or Band Pass filter
- Central Frequency offset (enter a value or by cursor selection)
- Averaging
- Down or Up Sampling (by ratio of an integer >1)
- Cut part of the signal in time (enter a value or by cursor selection)
- Save as a new .wav file
#### 3. Time Metrics
- Spectrogram (several variants, including 3D. STFT in Hann window : more options in future versions)
- Time/Amplitude
- Persistence Spectrum
- Phase Transitions
- Frequency Transitions
#### 4. Power Metrics
- Power Spectrum FFT (and variant)
- Signal power
- PSD
#### 5. Phase Metrics
- Constellation
- Phase Spectrum
- Phase Distribution (experimental ; to improve or remove later)
#### 6. Cyclostationarity Metrics
- Autocorrelation function (fast & complete)
#### 7. OFDM Metrics
- Estimation of : OFDM symbol duration, guard interval, subcarrier spacing

### Using the app
1. Simply clone/download the code in this repository to modify the code as needed for your purposes and run the main file to open the GUI.<br>
The GUI is in French by default, but can be switched to English (line 33, just swap "get_fra_lib" to "get_eng_lib").<br>
Debugging in the console can also be deactivated line 76.
```
python3 gui_main.py
```

2. Download and use the executable *SigAnto_v1.05.exe*. It is packaged in French but can still be swapped to English after launch in "Affichage/Switch language".
<br>
You can also easily package it yourself to share it to people unfamiliar with command line use : <br>
I would recommend the following pyinstaller command (having previously set up a virtual env with the files in this repo and only the required libraries), which would provide you with a directory containing an executable.

```
(python -m) pyinstaller --onedir --noconsole --icon=radio-waves.ico gui_main.py --name SigAnTo
```
Note that the executable provided in this repository was created with --onefile instead of --onedir : resulting in a single file obviously, but with slower start-up (it has to decompress in a temp directory at load).
The --noconsole argument with pyinstaller removes the console, which is not necessary for normal GUI use, but might trigger false antivirus flags during build.<br>

Packaging with Nuitka instead might lighten the size of the package and improve performance as well, but its ease of use is dependant upon OS, C compiler, python version... so I wouldn't recommend it unless you know what you are getting into.<br>

The requirements.txt contains the earliest tested versions ; the .exe provided here was packaged with python 3.13 and the latest stable versions of numpy, scipy and matplotlib so there should be no need to change you python environment if it is >3.8.<br>
The scipy dependency is responsible for roughly half the size of the app, so I'll remove that at some point by coding some of those functions myself. Matplotlib & Numpy will stay though.

### Examples :
- Spectrogram of a FSK17 signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_2.png" alt="FSK17"/>
- Constellation of a dPMR signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_3.png" alt="dPMR"/>
- PSD of a IS-95 (CDMA) signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_4.png" alt="WiFi"/>
- Symbol rate of a TETRA signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_5.png" alt="TETRA"/>
- Autocorrelation of a TETRAPOL signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_6.png" alt="TETRAPOL"/>
- Results of a CIS-45 OFDM signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_7.png" alt="CIS-45"/>
- Frequency transitions of a FSK signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_8.png" alt="FSK_transitions"/>

### Supported Hardware :
None ! As previously stated, this tool has no real-time applications.

### Useful resources :
- Information on signals and some example .wav files available : <br>
[Signal Identification Guide](https://www.sigidwiki.com/)<br>
[RadioReference Wiki](https://wiki.radioreference.com/index.php/)<br>
[Priyom.org](https://priyom.org/)
- [PySDR](https://pysdr.org/index.html) by Dr. Marc Lichtman

### Credits :
- Drs. FX Socheleau & S Houcke on OFDM blind analysis
- Dr. Marc Lichtman on Spectrogram generation