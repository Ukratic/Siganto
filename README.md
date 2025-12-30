## SigAnTo

Signal Analysis Toolbox : a free & open-source digital signal analyzer.<br>

Read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs, measurements and modifications. Some of the analysis is automated (see some examples with screenshots below) but should always be confirmed manually. 
FSK & PSK demodulation is also available, with more to come later.<br>
Real-time applications such as dealing with incoming streams are outside the scope of this tool.<br>

Be aware that the entire file is displayed, not looped over in chunks as many other SDR tools do, especially when they are meant to handle real-time display. Therefore, this tool is not meant to work with very long recordings. You may experience some acceptable sluggishness if your file contains samples in the order of a few millions, but the GUI will likely become downright painful to use with a file containing several tens or hundreds of millions of samples or more. Warnings are in place for a few potentially breaking actions with large files.<br>

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/>

### Available functions
#### 1. Display & Settings
- Load a .wav file. The button reloads the same file if clicked again. Select the button next to it to close the .wav file before loading another.<br>
Spectrogram & PSD shown when a file is loaded. With the V1.09 dependency, loading a wav file can be done by drag & drop.
- Activate measuring cursors (deactivated by default upon loading a new graph). Distance between 2 cursors is shown and you can also jump to a nearby peak (button in bottom right).
- Change language (English and French available)
- Display frequency & power information of the signal file (Estimated BW, dB level, symbol rate & ACF)
- Advanced Settings : Modify parameters not directly available on each graph/function.
#### 2. Signal modification
- Change FFT parameters :<br>
Window size (default based on number of samples), <br>
Window function (Kaiser, Hann, Hamming, Blackman, Bartlett, Flattop, Rectangular), <br>
Overlap.
- Central Frequency offset (enter a value, cursor selection or fine-tuning with arrow keys ; last option available only on tri-graph group for now)
- Coarse doppler correction (essentially a diagonal frequency shift)
- Down or Up Sampling (by ratio of an integer >1)
- Polyphase resampling (target sample rate)
- Cut part of the signal in time (enter a value or by cursor selection ; the latter only works reliably on spectrogram)
- Modify parameters for transition smoothing (phase, frequency) & persistence bins
- Save as a new .wav file
#### 4. Filters
- Low/High/Band Pass
- FIR, Wiener, Median, Moving average, Mean
- Matched Filter : Gaussian, Raised Cosine, Root Raised Cosine, Sinc, Rsinc, Rectangular
#### 4. Spectrograms
- Group : Spectrogram & DSP
- Group : Spectrogram, Constellation and Peak power spectrum. Frequency fine-tuning: Left/right arrows on keyboard for 1Hz step, clickable GUI arrows for 0.01Hz step
- 3D spectrogram
- Spectrogram
#### 5. Power Metrics
- Power Spectral Density
- Peak Power Spectrum
- Time Domain / Amplitude
#### 6. Frequency Metrics
- Persistence Spectrum
- Frequency Distribution
- Frequency Transitions
- Morlet CWT Scalogram
#### 7. Phase Metrics
- Constellation
- Phase Distribution
- Phase Transitions
- Eye Diagram (not synchronized)
#### 8. Symbol Rate Metrics
- Envelope Power Specturm
- Power Envelope Spectrum
- Clock Transition Spectrum
- Power Order (2 & 4) Spectrum
- Cyclospectrum
#### 9. Cyclostationarity Metrics
- Autocorrelation function (FFT-based fast variant or complete)
- Spectral Correlation function
#### 10. OFDM Metrics
- Estimation of : OFDM symbol duration, guard interval, subcarrier spacing
#### 11. Demodulation
- 2 or 4 CPM/FSK & PSK Demodulation (with optional baud rate estimation)
- AM, FM & SSB Demodulation (visual output and result signal saved for audio playback, further analysis or demodulation)
- MFSK Demodulation (2 methods : tone detection or from smoothed frequency transitions)
#### 12. Audio
- Audio playback (first pick the appropriate demodulation in the corresponding tab prior to audio playback, depending on how that signal was sourced and recorded)

### Examples :
- Spectrogram of a Chinese 4+4 <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_2.png" alt="Spectrogram"/>
- dPMR Constellation <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_3.png" alt="Constellation"/>
- PSD of CIS-FTM4 <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_4.png" alt="PSD"/>
- TETRA Symbol rate <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_5.png" alt="Symbol_rate"/>
- TETRAPOL Autocorrelation Function <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_6.png" alt="ACF"/>
- CIS-45 OFDM Parameters <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_7.png" alt="OFDM_params"/>
- EDACS Frequency transitions <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_8.png" alt="FSK_transitions"/>
- AIS Demodulation <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_9.png" alt="FSK_demod"/>

### Changelog :
- *V1.08* : <br>
-- More filtering options (median, moving average, gaussian, Wiener, FIR). Some tweaking needed to make them user-friendly.<br>
-- 4-FSK demodulation now actually functional.<br>
-- MFSK demodulation to handle multi-tone signals including non binary orders.<br>
-- Handling of various WAV encodings made more robust (PCM & IEEE float).<br>
-- Bug fixes on existing features (transitions smoothing & frequency distribution).<br>

- *V1.09* :<br>
-- Windowing options also used on PSD, phase & frequency transitions (previously only spectrogram). <br>
-- Streamlining filtering options and added Matched Filter.<br>
-- Added Cyclospectrum. Tweaked symbol rate estimation in the information window.<br>
-- Drag & drop files instead of opening a file dialog (optional feature adding the tkinterdnd2 library as a dependency).<br>
-- Eye Diagram added to Phase Metrics.<br>
-- Morlet CWT Scalogram added to Frequency Metrics.<br>
-- 2 & 4 PSK demodulation.<br>
-- Fixed MFSK demodulation.<br>
-- SCF in Cyclostationarity Metrics.<br>

- *V1.10* :<br>
-- Fixed power scaling on various graphs. Normalized when exact values are not relevant.<br>
-- Added polyphase resampling.<br>
-- Added Envelope Spectrum & renamed some symbol rate graphs with a clearer title.<br>
-- Automatic symbol rate estimation now has a confidence level.<br>

- *V1.10b-d* :<br>
-- Almost all parameters now accessible (options tab), no longer hardcoded.<br>
-- SSB demod .<br>

### Using the app
1. Clone/download the code in this repo to modify the code as needed for your purposes, run the main file *gui_main.py* to launch.<br>
Built on top of base Python (>=3.9 with Tkinter), Numpy, Matplotlib and Scipy. Sounddevice & tkinterdnd2 are optional.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university, industry or government environments.<br>

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