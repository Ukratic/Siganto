## SigAnTo

A lightweight signal analysis GUI toolbox.<br><br>

Its purpose is to read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs, measurements and modifications.
Real-time applications are outside of the scope of this tool.<br>
Some of the identification is automated (see some examples with screenshots below) but should always be confirmed manually.<br>

SigAnTo supports mostly analysis for now, with manual fine-tuning and 2 & 4-FSK demodulation. More to come later.<br>
Be aware that the entire file is displayed, not looped over as many other SDR tools do. Therefore, this tool is not meant to work with very long recordings, with which you would likely experience some acceptable sluggishness if your file contains samples in the order of a few millions but the GUI will become downright painful to use with a file containing several tens or hundreds of millions of samples or more.<br>

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/><br>

### Available functions <br>
#### 1. Display
- Load a .wav file. The button reloads the same file if clicked again. Select the button next to it to close the .wav file before loading another.<br>
Spectrogram shown when a file is loaded.
- Activate measuring cursors (deactivated by default upon loading a new graph). Distance between 2 cursors is shown and you can also jump to a nearby peak.
- Change FFT parameters :<br>
Window size (default based on number of samples), <br>
Window function (Kaiser, Hann, Hamming, Blackman, Bartlett, Flattop, Rectangular), <br>
Overlap.
- Change language (English and French available)
- Display frequency & power information of the signal file (Estimated BW, dB level, symbol rate & ACF)
- Modify parameters for transition (phase, frequency) & persistence graphs
#### 2. Signal modification
- Low, High or Band Pass filter
- Central Frequency offset (enter a value, cursor selection or fine-tuning with arrow keys ; last option available only on tri-graph group for now)
- Averaging (default mean level or defined value)
- Down or Up Sampling (by ratio of an integer >1)
- Cut part of the signal in time (enter a value or by cursor selection ; the latter only works reliably on spectrogram)
- Save as a new .wav file
#### 3. Main graphs
- Spectrogram (STFT & 3D)
- Groups with several graphs on the same window
#### 4. Power Metrics
- Power Spectrum FFT (and variant)
- Signal power
- PSD (and variant)
- Time/Amplitude (IQ samples)
#### 5. Frequency Metrics
- Persistence Spectrum
- Frequency Distribution
- Frequency Transitions
#### 6. Phase Metrics
- Constellation
- Phase Spectrum
- Phase Distribution
- Phase Transitions
#### 7. Cyclostationarity Metrics
- Autocorrelation function (fft-based fast variant or complete)
#### 8. OFDM Metrics
- Estimation of : OFDM symbol duration, guard interval, subcarrier spacing
#### 9. Demodulation
- 2 & 4 FSK Demodulation

### Examples :
- Spectrogram of a Chinese 4+4 signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_2.png" alt="4_4"/>
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
- Frequency transitions of an AIS signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_8.png" alt="FSK_transitions"/>
- Demodulation of an EDACS signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_9.png" alt="FSK_demod"/>

### Changelog :
- December 2024 : <br>
-- First upload on github<br>
-- Fixed language output in the console, now correctly dependent on language choice in the GUI (except debug output).<br>
-- Executable *SigAnto_v1.05.exe* available.

- January 2025 : <br>
-- Added option to choose the window function of the STFT : Hann, Hamming, Blackman, Bartlett, Kaiser, Flat Top, Rectangular.<br>
-- Added dynamic frequency resolution on file load instead of fixed FFT size, hopefully improving first look at a signal.<br>
-- Streamlining of spectrogram options and display.<br>
-- Testing new feature : Arrows for frequency fine-tuning (1Hz step). Only on Spectrogram/Constellation group for now.<br> 

- February 2025 : <br>
-- Testing new feature : 2 & 4 FSK demodulation.<br>
-- Executable *SigAnto_v1.06.exe* available.

- In future versions : <br>
-- Automatic modulation recognition, if I can figure out a decent algorithm.


### Using the app
1. Simply clone/download the code in this repository to modify the code as needed for your purposes and run the main file *gui_main.py* to launch.<br>
The GUI is in French by default, but can easily be switched to English (line 37, just swap "get_fra_lib" to "get_eng_lib"). Debugging in the console can also be deactivated line 84.<br>
It is built on top of base python (>=3.9 with tkinter), numpy, matplotlib and scipy.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university, industry or government environments (if not, use the executable to avoid dependencies altogether).
The requirements.txt contains the earliest tested versions ; the .exe provided here was packaged with python 3.13 and the latest stable versions of numpy, scipy and matplotlib so there should be no need to change your environment to run the code if you are using a python version equal or above 3.8.<br>

2. Download and use the executable *SigAnto_v1.06.exe* (french) or *SigAnto_v1.06_eng.exe* from the releases tab.<br>
Languages can still be swapped after launch in the display options.

### Supported Hardware :
None ! As previously stated, this tool has no real-time applications and can only work with .wav recordings.

### Supported file format :
WAV 8-bit, 16-bit, 32-bit, 64-bit.<br>
Scripts to convert from SigMF & MP3 are available in this repo.<br>
SigMF conversion should work without a hitch though the metadata won't make it into the .wav file and therefore won't be shown in the GUI. 
MP3 conversion requires ffmpeg and might also be a little more sketchy depending on how the file was recorded & encoded ; the result should be considered carefully.

### Useful resources :
- Information on signals and some example .wav files available : <br>
[Signal Identification Guide](https://www.sigidwiki.com/)<br>
[RadioReference Wiki](https://wiki.radioreference.com/index.php/)<br>
[Priyom.org](https://priyom.org/)<br>
[Tony Anselmi's blog](https://i56578-swl.blogspot.com/)

### Credits :
- Drs. FX Socheleau & S Houcke on OFDM blind analysis
- Dr. Marc Lichtman on Spectrogram generation and a great starter for python DSP : [PySDR](https://pysdr.org/index.html)
- Michael Ossmann on Clock Recovery