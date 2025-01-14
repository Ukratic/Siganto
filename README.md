## SigAnTo

A lightweight signal analysis toolbox GUI.<br><br>
Built on top of base python (>=3.9 with tkinter), numpy, matplotlib and scipy.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university or industry environments (or use the executable to avoid dependencies altogether).

Its purpose is to read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs and measurements.
Real-time applications are outside of the scope of this tool.<br>
Some of the identification is automated (see some examples with screenshots below) but should always be confirmed manually.<br>

The tool only supports analysis for now, fine-tuning and demodulation will be covered in future versions.<br>
At some point I might also switch from Tkinter to PyQt for better performance ; be aware that in the current version, you will likely experience some acceptable sluggishness if your file contains samples in the order of a few millions but the GUI will become downright painful to use with a file containing several tens or hundreds of millions of samples or more.<br>

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/><br>

### Available functions <br>
#### 1. Display
- Load a .wav file. The button reloads the same file if clicked again. Select the button next to it to close the .wav file before loading another.<br>
Spectrogram shown when a file is loaded.
- Activate measuring cursors (deactivated by default upon loading a new graph). Distance between 2 cursors is shown and you can also jump to a nearby peak.
- Change FFT parameters :<br>
Window size (default based on number of samples), <br>
Window function (Tukey, Kaiser, Hann, Hamming, Blackman, Bartlett, Flattop, Rectangular), <br>
Overlap.
- Change language (English and French available)
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
- Spectrogram (STFT & 3D)
- Time/Amplitude (IQ samples)
- Persistence Spectrum
- Phase Transitions
- Frequency Transitions
#### 4. Power Metrics
- Power Spectrum FFT (and variant)
- Signal power
- PSD (and variant)
#### 5. Phase Metrics
- Constellation
- Phase Spectrum
- Phase Distribution
#### 6. Cyclostationarity Metrics
- Autocorrelation function (fft-based fast variant or complete)
#### 7. OFDM Metrics
- Estimation of : OFDM symbol duration, guard interval, subcarrier spacing

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
- Frequency transitions of a FHSS/FSK17 signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_8.png" alt="FSK_transitions"/>

### Changelog :
- December 2024 : <br>
-- First upload on github<br>
-- Fixed language output in the console, now correctly dependent on language choice in the GUI.<br>
-- Executable *SigAnto_v1.05.exe* available.

- January 2025 : <br>
-- Added option to choose the window function of the STFT : Hann, Hamming, Blackman, Bartlett, Kaiser, Flat Top, Rectangular.<br>
-- Added dynamic frequency resolution on file load instead of fixed FFT size, improving first look at a signal.<br>
-- Streamlining of spectrogram options and display.<br>
-- Testing new feature : Slider for frequency fine-tuning. Only on Spectrogram/Constellation group for now.

### Using the app
1. Simply clone/download the code in this repository to modify the code as needed for your purposes and run the main file to open the GUI.<br>
The GUI is in French by default, but can be switched to English (line 34, just swap "get_fra_lib" to "get_eng_lib").<br>
Debugging in the console can also be deactivated line 80.
```
python3 gui_main.py
```

2. Download and use the executable *SigAnto_v1.051.exe* (from 12th of January 2024 code).
It is packaged in French but can still be swapped to English after launch in "Affichage/Switch language".
<br>
You can also easily package it yourself to share it to people unfamiliar with command line use : <br>
I would recommend the following pyinstaller command (having previously set up a virtual env with the files in this repo and only the required libraries), which would provide you with a directory containing an executable.

```
(python -m) pyinstaller --onedir --noconsole --icon=radio-waves.ico gui_main.py --name SigAnTo
```
Note that the executable provided in this repository was created with --onefile instead of --onedir : resulting in a single file obviously, but with slower start-up (it has to decompress in a temp directory at load).
The --noconsole argument with pyinstaller removes the console, which is not necessary for normal GUI use, but might trigger false antivirus flags during build.<br>

Packaging with Nuitka instead might lighten the size of the package and improve performance as well, but its ease of use is dependant upon OS, C compiler, python version... so I wouldn't recommend it unless you know what you are getting into.<br>

The requirements.txt contains the earliest tested versions ; the .exe provided here was packaged with python 3.13 and the latest stable versions of numpy, scipy and matplotlib so there should be no need to change your python environment if it is in a version > 3.8.<br>
The Scipy dependency is responsible for roughly half the size of the packaged app (contains Scipy.stats and lots of stuff that I am not using), so I might one day make the effort to remove that by coding some of those functions myself... but this is a not unsignificant effort that I am not sure would be worth it since Scipy should be considered a standard almost everywhere. Numpy on the other hand will most definitely stay ; Matplotlib will as well, at least until (if ever) I move to another GUI library.

### Supported Hardware :
None ! As previously stated, this tool has no real-time applications.

### Supported file format :
WAV 8-bit, 16-bit, 32-bit, 64-bit.<br>
Scripts to convert from SigMF & mp3 available in this repo (SigMF should work without a hitch though you'll of course lose the metadata. Mp3 conversion might be a little more sketchy and the result should be considered carefully) and requires ffmpeg.

### Useful resources :
- Information on signals and some example .wav files available : <br>
[Signal Identification Guide](https://www.sigidwiki.com/)<br>
[RadioReference Wiki](https://wiki.radioreference.com/index.php/)<br>
[Priyom.org](https://priyom.org/)

### Credits :
- Drs. FX Socheleau & S Houcke on OFDM blind analysis
- Dr. Marc Lichtman on Spectrogram generation and a great starter for python DSP : [PySDR](https://pysdr.org/index.html) 