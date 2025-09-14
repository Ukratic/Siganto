## SigAnTo

Signal Analysis Toolbox : a free & open-source digital signal analyzer.<br>

Read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs, measurements and modifications.
Real-time applications such as dealing with incoming streams are outside the scope of this tool.<br>
Some of the identification is automated (see some examples with screenshots below) but should always be confirmed manually.<br>

SigAnTo supports mostly analysis for now, with manual fine-tuning and 2 & 4-FSK demodulation. More to come later.<br>
Be aware that the entire file is displayed, not looped over in chunks as many other SDR tools do (with real-time display). Therefore, this tool is not meant to work with very long recordings, you may experience some acceptable sluggishness if your file contains samples in the order of a few millions but the GUI will likely become downright painful to use with a file containing several tens or hundreds of millions of samples or more.<br>

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/>

### Available functions
#### 1. Display
- Load a .wav file. The button reloads the same file if clicked again. Select the button next to it to close the .wav file before loading another.<br>
Spectrogram & PSD shown when a file is loaded. With the V1.09 dependency, loading a wav file can be done by drag & drop.
- Activate measuring cursors (deactivated by default upon loading a new graph). Distance between 2 cursors is shown and you can also jump to a nearby peak (button in bottom right).
- Change language (English and French available)
- Display frequency & power information of the signal file (Estimated BW, dB level, symbol rate & ACF)
#### 2. Signal modification
- Change FFT parameters :<br>
Window size (default based on number of samples), <br>
Window function (Kaiser, Hann, Hamming, Blackman, Bartlett, Flattop, Rectangular), <br>
Overlap.
- Central Frequency offset (enter a value, cursor selection or fine-tuning with arrow keys ; last option available only on tri-graph group for now)
- Down or Up Sampling (by ratio of an integer >1)
- Cut part of the signal in time (enter a value or by cursor selection ; the latter only works reliably on spectrogram)
- Modify parameters for transitions (phase, frequency) & persistence graphs
- Save as a new .wav file
#### 4. Filters
- Low/High/Band Pass
- FIR, Wiener, Median, Moving average, Mean
- Matched Filter : Gaussian, Raised Cosine, Root Raised Cosine, Sinc, Rsinc, Rectangular
#### 4. Main graphs
- Spectrogram (STFT & 3D)
- Groups combining graphs on the same window
#### 5. Power Metrics
- Power Spectrum FFT (and variant)
- Signal power FFT
- Cyclospectrum FFT
- PSD (and variant)
- Time/Amplitude (IQ samples)
#### 6. Frequency Metrics
- Persistence Spectrum
- Frequency Distribution
- Frequency Transitions
- Morlet CWT Scalogram
#### 7. Phase Metrics
- Constellation
- Phase Spectrum
- Phase Distribution
- Phase Transitions
- Eye Diagram
#### 8. Cyclostationarity Metrics
- Autocorrelation function (fft-based fast variant or complete)
#### 9. OFDM Metrics
- Estimation of : OFDM symbol duration, guard interval, subcarrier spacing
#### 10. Demodulation
- 2 or 4 CPM/FSK & PSK Demodulation (outputs a stream of bits and graph of transitions)
- AM & FM Demodulation (visual output and result signal saved for playback or further analysis)
- MFSK Demodulation (2 methods : tone detection or from smoothed frequency transitions)
#### 11. Audio
- Audio playback (when listening to an amplitude modulated signal for instance, you may need to use the appropriate demodulation in the corresponding tab prior to audio playback, depending on how that signal was sourced and recorded)

### Examples :
- Spectrogram of a Chinese 4+4 <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_2.png" alt="4_4"/>
- dPMR Constellation <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_3.png" alt="dPMR"/>
- PSD of IS-95 (CDMA) <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_4.png" alt="WiFi"/>
- TETRA Symbol rate <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_5.png" alt="TETRA"/>
- TETRAPOL Autocorrelation Function <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_6.png" alt="TETRAPOL"/>
- CIS-45 OFDM Parameters <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_7.png" alt="CIS-45"/>
- AIS Frequency transitions <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_8.png" alt="FSK_transitions"/>
- EDACS Demodulation <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_9.png" alt="FSK_demod"/>

### Changelog :
- *V1.05* : <br>
-- First upload on github<br>

- *V1.06* : <br>
-- Options for the window function of the STFT : Hann, Hamming, Blackman, Bartlett, Kaiser, Flat Top, Rectangular.<br>
-- Dynamic frequency resolution on file load instead of fixed FFT size, improving first look at a signal in most cases.<br>
-- Arrows for frequency fine-tuning (1Hz step). Only on Spectrogram/Constellation group for now.<br> 
-- 2 & 4 FSK demodulation.<br>
-- Bug fixes on existing features.<br>

- *V1.07* : <br>
-- AM/FM demodulation.<br>
-- Audio Output (optional feature adding the sounddevice library as a dependency).<br>
-- Bug fixes on existing features.<br>

- *V1.08* : <br>
-- More filtering options (median, moving average, gaussian, Wiener, FIR). Some tweaking needed to make them user-friendly.<br>
-- 4-FSK demodulation now actually functional.<br>
-- MFSK demodulation to handle multi-tone signals including non binary orders.<br>
-- Handling of various WAV encodings made more robust (PCM & IEEE float).<br>
-- Bug fixes on existing features (transitions smoothing & frequency distribution).<br>

- *V1.09* :<br>
-- Windowing options also used on PSD, phase & frequency transitions (previously only spectrogram). <br>
-- Streamlining filtering options and added Matched Filter.<br>
-- Cyclospectrum added to Power Metrics. Expanded symbol rate estimation in the information window.<br>
-- Drag & drop files instead of opening a file dialog (optional feature adding the tkinterdnd2 library as a dependency).<br>
-- Eye Diagram added to Phase Metrics.<br>
-- Morlet CWT Scalogram added to Frequency Metrics.<br>
-- 2 & 4 PSK demodulation.<br>

*Planned for later :*<br>
-- Cursor selection improvement.<br>
-- More parameters available in GUI.<br>
-- More demodulation options (differential, offset...).<br>


### Using the app
1. Simply clone/download the code in this repository to modify the code as needed for your purposes and run the main file *gui_main.py* to launch.<br>
It is built on top of base Python (>=3.9 with Tkinter), Numpy, Matplotlib and Scipy. Sounddevice & tkinterdnd2 are optional.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university, industry or government environments (if not, use the executable to avoid dependencies altogether).
The requirements.txt contains the earliest tested versions ; the .exe provided here was packaged with Python 3.13 and the latest stable versions of numpy, scipy and matplotlib so there should be no need to change your environment to run the code if you are using a Python version equal or above 3.9.<br>

2. Download and use the executable from the releases tab (language can still be swapped after launch).

### Supported file format :
WAV 8-bit, 16-bit, 32-bit, 64-bit, encoding PCM & IEEE floats, IQ or real.<br>
Scripts to convert from SigMF & MP3 are available in this repo.<br>
SigMF conversion should work fine (though not maintained if the format evolves) but most of the metadata will be lost.
MP3 conversion requires ffmpeg and might also be a little sketchy depending on how the file was recorded & encoded ; the result should be considered carefully.

### Useful resources :
- Information on signals and some example .wav files and/or analysis available : <br>
[Signal Identification Guide](https://www.sigidwiki.com/)<br>
[RadioReference Wiki](https://wiki.radioreference.com/index.php/)<br>
[Priyom.org](https://priyom.org/)<br>
[Tony Anselmi's blog](https://i56578-swl.blogspot.com/)

### Credits :
- Drs. FX Socheleau & S Houcke on OFDM blind analysis
- Dr. M Lichtman on a great starter for Python DSP : [PySDR](https://pysdr.org/index.html)
- Michael Ossmann on Clock Recovery