## SigAnTo

Signal Analysis Toolbox : a free & open-source digital signal analyzer.<br>

Read .wav files (recorded from SDR tools, obtained from somewhere, simulated...) and identify the signal parameters (modulation scheme, symbol rate, ACF...) through various graphs, measurements and modifications. Some of the analysis is automated (see some examples with screenshots below) but should always be confirmed manually. FSK & PSK demodulation is also available.
Real-time applications such as dealing with incoming streams are outside the scope of this tool.<br>

The entire file is displayed, not looped over in chunks as many other SDR tools do, especially when they are meant to handle real-time display. 
Consequently, SigAnTo can handle relatively large files (tens of millions of samples) without significant drawbacks but isn't equipped to smoothly deal with multi-gigabyte files.<br>
Some functions are computationally very demanding and should not be used with large files, plotting the SCF for instance. In those cases, splitting the file into smaller chunks before analysis is recommended. Warnings are in place for those functions that may crash with large files.<br>

<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_1.png" alt="Main"/>

### Available functions
#### Bottom buttons
- Load a .wav file. The button reloads the same file if clicked again. Select the button next to it to close the .wav file before loading another.<br>
Spectrogram & PSD are shown when a file is loaded. With the added dependency of the tkinterdnd2 library, loading a wav file can be done by drag & drop.
- Activate measuring cursors (deactivated by default upon loading a new graph). Distance between 2 cursors is shown when a second one is placed.
- Jump to nearest peak (from the latest cursor positioned on the graph)

#### 1. Display & Settings
- Change language (English and French available)
- Display frequency & power information of the signal file (Estimated BW, dB level, estimated symbol rate & ACF)
- Advanced Settings : Modify parameters not directly available on each graph/function
- Load another type (non-wav) of file
- Show wav metadata

#### 2. Signal modification
- Manual Central Frequency offset (enter a value, cursor selection or fine-tuning with arrow keys ; last option available only on tri-graph group for now)
- Automatic Central Frequency Centering (modulation-based)
- Coarse doppler correction (essentially a diagonal frequency shift)
- Resampling : either Polyphase or coarse Down/Up Sampling by ratio of an integer >1
- Polyphase resampling (target sample rate)
- Cut part of the signal in time alone or time & frequency (enter a value or by cursor selection ; the latter only works reliably on spectrogram)
- Change FFT parameters :<br>
Window size (default based on number of samples), <br>
Window function (Kaiser, Hann, Hamming, Blackman, Bartlett, Flattop, Blackman-Harris 7-term, Gaussian, Lanczos, Rectangular), <br>
Overlap.
- Additional options : Modify parameters for transition smoothing (phase, frequency) & persistence bins, Correct Constellation rotation
- Save as a new .wav file (mono or stereo)
- Split a wav into smaller parts
#### 3. Filters
- Low-pass, High-pass, Band-pass, Stop-band
- Spectral denoising
- FIR, Wiener, Median, Moving average
- Matched Filter : Gaussian, Raised Cosine, Root Raised Cosine, Sinc, Rsinc, Rectangular
#### 4. Spectrograms
- Group : Spectrogram & DSP
- Group : Spectrogram, Constellation and Instantaneous Phase. Frequency fine-tuning: Left/right arrows on keyboard for 1Hz step, clickable GUI arrows for 0.01Hz step
- 3D spectrogram
- Spectrogram
#### 5. Power Metrics
- Power Spectral Density
- Peak Power Spectrum
- Time Domain / Amplitude
- Signal at nth power
#### 6. Frequency Metrics
- Persistence Spectrum
- Frequency Distribution
- Frequency Transitions
#### 7. Phase Metrics
- Constellation
- Phase Distribution
- Phase Transitions
- Eye Diagram
- Synchronized Constellation (at symbol rate)
#### 8. Symbol Rate Metrics
- Power Envelope Spectrum
- Clock Transition Spectrum
- Power Order (2 & 4) Spectrum
- Cyclospectrum
- Clock Frequency Detection
#### 9. Cyclostationarity Metrics
- Autocorrelation function (FFT-based fast variant or complete)
- Spectral Correlation function
#### 10. OFDM Metrics
- Two-step estimation of : OFDM symbol duration, guard interval, subcarrier spacing.
#### 11. Demodulation
- 2 & 4 CPM/FSK or PSK Demodulation (with optional baud rate estimation)
- AM, FM & SSB Demodulation (visual output and result signal saved for audio playback, further analysis or demodulation)
- MFSK Demodulation (frequency tones automatically determined or from manual input)
#### 12. Audio
- Audio playback (first pick the appropriate demodulation in the corresponding tab prior to audio playback, depending on how that signal was sourced and recorded). Requires the sounddevice library.
#### 13. Experimental
- Morlet CWT Scalogram
- Cepstral Estimator
- Bispectrum
- LoRa score

### Using the app
1. Run the main file *gui_main.py* to launch.<br>
Built on top of base Python (>=3.9 with Tkinter), Numpy, Matplotlib and Scipy. Sounddevice & tkinterdnd2 are optional.<br>
Its low list of dependencies will hopefully allow most people to run it without too much hassle in university, industry or government environments.<br>

2. Download and use the executable from the last version in the releases tab (language can still be swapped after launch).

### Supported file format :
WAV 8-bit, 16-bit, 32-bit, 64-bit, encoding PCM & IEEE floats, IQ or real.<br>

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

- EDACS Instantaneous Frequency <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_8.png" alt="Instantaneous Frequency"/>

- AIS Demodulation <br>
<img src="https://github.com/Ukratic/Siganto/blob/main/images/pic_9.png" alt="FSK_demod"/>


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