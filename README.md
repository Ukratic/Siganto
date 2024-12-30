## SigAnTo

A signal analysis toolbox GUI built in python<br>
Built on top of base python (>3.9 with tkinter), numpy, matplotlib and scipy.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university or industry environments (or use the executable to avoid dependencies altogether).

Its purpose is to read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs and measurements.
Real-time applications are outside of the scope of this tool.<br>
Some of the identification is automated (see examples further down) but should always be confirmed manually.<br>

The tool only supports analysis for now, fine-tuning and demodulation will be covered in further versions.<br>
In this next stage, I might also switch from Tkinter to PyQt for better performance ; be aware that you will likely experience some sluggishness if your file contains samples in the order of a few millions and the GUI will become downright painful to use with a file containing several tens of millions of samples or more.

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/><br>

Usage : <br>
1. Simply clone/download the code in this repository to modify the code as needed for your purposes and run the main file to open the GUI.<br>
The GUI is in French by default, but can be switched to English (line 33, just swap "get_fra_lib" to "get_eng_lib").<br>
Debugging in the console can also be deactivated line 76.
```
python3 gui_main.py
```

2. Use the SigAnTo executable. It is packaged (see further down how to do this yourself) in French but can still be swapped to English after launch in "Affichage/Switch language".

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
The repository also contains 2 scripts to convert mp3 and SigMF files into wav, so it can then be read into SigAnTo (or whatever else shares .wav requirements). Be aware the mp3 conversion requires ffmpeg.
<br><br>
The code is easily packageable for sharing purposes to people unfamiliar with command line use : <br>
The following pyinstaller command (having previously set up a virtual env with the code and only the required libraries) will provide you with an executable.<br>

```
pyinstaller --onefile --icon=radio-waves.ico gui_main.py --name SigAnTo
```
Note that with pyinstaller, the --noconsole argument would remove the console which is not necessary for normal GUI use, but this might trigger false antivirus flags.
Packaging with Nuitka instead might provide a solution to this, TBD.

### Examples :
- Spectrogram of a FSK17 signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_2.png" alt="FSK17"/>
- Constellation of a dPMR signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_3.png" alt="dPMR"/>
- PSD of a WiFi signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_4.png" alt="WiFi"/>
- Symbol rate of a TETRA signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_5.png" alt="TETRA"/>
- Autocorrelation of a TETRAPOL signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_6.png" alt="TETRAPOL"/>
- Results of a CIS-45 OFDM signal <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_7.png" alt="CIS-45"/>

### Useful resources :
- Information on signals and some example .wav files available : <br>
[Signal Identification Guide](https://www.sigidwiki.com/)<br>
[RadioReference Wiki](https://wiki.radioreference.com/index.php/)<br>
[Priyom.org](https://priyom.org/)
- [PySDR](https://pysdr.org/index.html) by Dr. Marc Lichtman

### Credits :
- Drs. FX Socheleau & S Houcke on OFDM blind analysis
- Dr. March Lichtman on Spectrogram generation