## SigAnTo

A lightweight signal analysis toolbox GUI built in python<br>
Built on top of base python (>3.9 with tkinter), numpy, matplotlib and scipy.<br>

Its purpose is to read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation, symbol rate, ACF...) through various graphs and measurements.
Real-time applications are outside of the scope of this tool.<br>
Some of this is automated (see examples further down) but should always be confirmed manually.<br>

The tool only supports analysis for now, fine-tuning and demodulation will be covered in further versions.<br>
At this stage I might also switch from Tkinter to PyQt for better performance ; be aware that currently you will likely experience some delay if your file contains samples in the order of a few millions and the GUI will become downright painful to use with a file containing tens of millions of samples or more.

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main" width="600" height="400"/><br>

The GUI is in French by default, but can be switched to English (line 33, just swap "get_fra_lib" to "get_eng_lib").
<br>
Usage : simply clone/download the repo and run the main file to open the GUI:

```
python3 gui_main.py
```

### Summary of available functions <br>
#### 1. Display
- Load .wav file
- Activate measuring cursors
- Change FFT window size
- Change language (english and french available)
- Display frequency & power information of the signal file
- Modify parameters for transition and persistence graphs
#### 2. Signal modification
- Low, High or Band Pass filter
- Central Frequency move (value or cursor selection)
- Averaging
- Down or Up Sampling
- Cut part of the signal in time (value or cursor selection)
- Save as new .wav file
#### 3. Time Metrics
- Spectrogram (several variants)
- Time/Amplitude
- Persistence Spectrum
- Phase Transitions
- Frequency Transitions
#### 4. Power Metrics
- Power Spectrum FFT (several variants)
- PSD
#### 5. Phase Metrics
- Constellation
- Phase Spectrum
- Phase Distribution
#### 6. Cyclostationarity Metrics
- Autocorrelation function (fast & complete)
#### 7. OFDM Metrics
- Estimation of : OFDM symbol duration, guard interval, subcarrier spacing

<br>
Expected encoding is standard 16bit wav, but 8 to 64bit is supported. Other formats might be added if there is a need for it.
<br><br>
The repo also contains 2 scripts to convert mp3 and SigMF files into wav, so it can then be read into SigAnTo (or whatever else shares .wav requirements). Be aware the mp3 conversion requires ffmpeg.
<br><br>
The code is easily packageable for sharing purposes, even to people unfamiliar with command line use : <br>
The following pyinstaller command (having previously set up a virtual env with the code and only the required libraries) will provide you with an executable.<br>

```
pyinstaller --onefile --icon=radio-waves.ico gui_main.py --name SigAnTo
```

The --onefile argument removes the console, which is not needed for normal use of the GUI ; it just provides a history of executed functions and can be useful for debugging.

### Examples :
- Spectrogram of a FSK17 signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_2.png" alt="FSK17" width="600" height="400"/>
- Constellation of a dPMR signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_3.png" alt="dPMR" width="600" height="400"/>
- PSD of a WiFi signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_4.png" alt="WiFi" width="600" height="400"/>
- Symbol rate of a TETRA signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_5.png" alt="TETRA" width="600" height="400"/>
- Autocorrelation of a TETRAPOL signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_6.png" alt="TETRAPOL" width="600" height="400"/>
- Results of a CIS-45 OFDM signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_7.png" alt="CIS-45" width="600" height="400"/>
