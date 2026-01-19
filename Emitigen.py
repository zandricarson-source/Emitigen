"""
SETUP:
Use: generates a sound frequency and amplitude scaled from the wavelength and relative intensity of an element
current final output is table of Å, Rel Intensity, Hz, Rel Amplitude, (R, G, B)
1. NIST ASD data (input element='H') ion stage default I, wavelength range default (3800, 7500)
    1.1 parse table to get wavelength (Å) and RI (excluding where RI is N/A)
    1.2 return Å, RI table
2. new table with sound data
    2.1 convert Å to Hz scaled from 3800-7500 to 20.0, 20000.0
    2.2 convert RI to floating point amplitude
        2.2.1 scale RI range from 0 to 10000 to range of 0 to 1 
    2.3 return Hz, Amp table
3. convert Å to RGB
    3.1 use wavelength_to_rgb(Å/10)
4. convert list of sound frequencies and amplitudes into a sound
    4.1 use scipy wave file
5. display the color and sound somehow
    5.1 just json to port it to javascript, visualize there
"""

from astroquery.nist import Nist
import astropy.units as u
import numpy as np
from scipy.io import wavfile
import json

VISIBLE_RANGE = (3800, 7500)

def get_element_spectrum(element='H'):
    """
    Docstring for get_element_table
    
    :param element: Element key, default is Hydrogen
    :return: a table of A wavelength, RE relative intensity
    """
    ion_stage='I'
    wavelength_range=VISIBLE_RANGE

    linename = f"{element} {ion_stage}"

    try:
        if wavelength_range:
            # Query with wavelength range
            min_wl, max_wl = wavelength_range
            table = Nist.query(min_wl * u.AA, max_wl * u.AA, linename=linename)
        else:
            # Query all wavelengths (use a very broad range)
            table = Nist.query(1 * u.AA, 1000000 * u.AA, linename=linename)
        
        if table is None or len(table) == 0:
            print(f"No data found for {linename}")
            return None, None, None
        
        # Extract Ritz wavelengths (calculated from energy levels)
        # Column name is 'Ritz' in the table
        ritz_wl = []
        rel_int = []
        for row in table:
            # Get Ritz wavelength (prioritize this over observed)
            if 'Ritz' in table.colnames and row['Ritz']:
                try:
                    # Clean the wavelength string (remove +, ?, *, etc.)
                    wl_str = str(row['Ritz']).strip()
                    # Remove common NIST quality indicators
                    wl_str = wl_str.rstrip('+?*:')
                    wl = float(wl_str)
                    ritz_wl.append(wl)
                    
                    # Get relative intensity if available
                    if 'Rel.' in table.colnames and row['Rel.']:
                        try:
                            int_str = str(row['Rel.']).strip()
                            int_str = int_str.rstrip('+?*:')
                            intensity = float(int_str)
                        except (ValueError, TypeError):
                            intensity = np.nan
                    else:
                        intensity = np.nan
                    rel_int.append(intensity)
                    
                except (ValueError, TypeError):
                    # Skip rows that can't be converted
                    continue
        wavelengths = np.array(ritz_wl)
        intensities = np.array(rel_int)

        valid_mask = ~np.isnan(intensities)
        wavelengths = wavelengths[valid_mask]
        intensities = intensities[valid_mask]
        
        print(f"Retrieved {len(wavelengths)} spectral lines for {linename}")
        if len(wavelengths) > 0:
            print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} Å")
        
        return wavelengths.tolist(), intensities.tolist()
    
    except Exception as e:
        print(f"Error querying NIST: {e}")
        print("Make sure you have astroquery installed: pip install astroquery")
        return None, None

# helper function, gamma in our case is the absolue amplitude generated from the relative intensity
# from github: https://gist.github.com/error454/65d7f392e1acd4a782fc
def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))

#helper function, wavelength to sound requency
def A_to_hz_log(A):
    A = max(3800, min(7500, A))

    wl_min, wl_max = 3800, 7500
    f_min, f_max = 20.0, 20000.0

    x = (wl_max - A) / (wl_max - wl_min)

    return float(f_min * (f_max / f_min) ** x)

#helper function
def RI_to_Amp(ri, max_ri):
    """
    Docstring for RI_to_Amp
    
    :param ri: an RI value
    :param max_ri: the largest value in the set
    :return amp: returns the amplitude float 0-1
    """
    amp = ri/max_ri
    return float(amp)

# generates a sound file in order to hear the element's frequency data
def freq_and_amp_to_sound(frequencies, amplitudes):
    frequencies = np.array(frequencies)  # Hz
    amplitudes = np.array(amplitudes)    # Relative amplitudes

    # Audio parameters
    sample_rate = 44100  # CD quality
    duration = 20.0       # seconds

    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate composite waveform by summing all frequency components
    composite_wave = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        composite_wave += amp * np.sin(2 * np.pi * freq * t)
    
    # Normalize to prevent clipping
    composite_wave = composite_wave / np.max(np.abs(composite_wave))
    # Convert to 16-bit PCM
    audio_data = np.int16(composite_wave * 32767)

    # Save to file
    wavfile.write('composite_sound.wav', sample_rate, audio_data)
    print("file saved")

    return audio_data

# element -> color and sound data
def get_color_and_sound_data(element):
    """
    Docstring for get_color_and_sound_data
    
    :param element: element symbol
    : output: returns a table of Å, Rel Intensity, Hz, Rel Amplitude, (R, G, B)
    """
    wavelengths, rel_intensities = get_element_spectrum(element)
    
    if wavelengths is None or rel_intensities is None:
        print(f"No valid spectrum data for {element}, skipping...")
        return None, None, None, None, None
    
    if len(wavelengths) == 0:
        print(f"Empty spectrum data for {element}, skipping...")
        return None, None, None, None, None
    
    frequencies = []
    amplitudes = []
    rgbs = []
    valid_wavelengths = []
    valid_intensities = [] # removes entries where the rgb value is 0, 0 ,0

    for i in range(len(rel_intensities)):
        freq = A_to_hz_log(wavelengths[i])
        amp = RI_to_Amp(rel_intensities[i], max(rel_intensities))
        rgb = wavelength_to_rgb(wavelengths[i]/10, rel_intensities[i]/max(rel_intensities))

        #filter (0, 0, 0) colors out
        if rgb != (0, 0, 0):
            valid_wavelengths.append(wavelengths[i])
            valid_intensities.append(rel_intensities[i])
            frequencies.append(freq)
            amplitudes.append(amp)
            rgbs.append(rgb)

    if len(valid_wavelengths) == 0:
        print(f"No visible spectrum data for {element}, skipping...")
        return None, None, None, None, None
    
    #debug
    #print(f"wave len: {len(wavelengths)} inten len: {len(rel_intensities)} freq len: {len(frequencies)} amp len: {len(amplitudes)} rgb len: {len(rgb)}")
    return valid_wavelengths, valid_intensities, frequencies, amplitudes, rgbs
    

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ELEMENT SPECTRUM CONVERTER")
    print("=" * 70)

    #example
    #wavelengths, rel_intensities, frequencies, amplitudes, rgb = get_color_and_sound_data('H')
    #audio = freq_and_amp_to_sound(frequencies, amplitudes)

    symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
                'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 
                'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 
                'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 
                'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
                'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
                'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
                'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 
                'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 
                'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    data = {}

    #generate data for each element into json
    for symbol in symbols:
        wavelengths, rel_intensities, frequencies, amplitudes, rgb = get_color_and_sound_data(symbol)
        if wavelengths is None:
            print(f"Skipping {symbol} - no spectral data available")
            data[symbol] = {
                "symbol": symbol,
                "wavelengths": None,
                "rel_intensities": None,
                "frequencies": None,
                "amplitudes": None,
                "colors": None
            }
            continue

        data[symbol] = {
            "symbol": symbol,
            "wavelengths": wavelengths,
            "rel_intensities": rel_intensities,
            "frequencies": frequencies,
            "amplitudes": amplitudes,
            "colors": rgb
        }
    
    #export to json
    with open('elements.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("exported data to elements.json")
    

    #audio_data = freq_and_amp_to_sound(table[0], table[1])