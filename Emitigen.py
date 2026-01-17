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

"""

from astroquery.nist import Nist
import astropy.units as u
import numpy as np

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
        
        return wavelengths, intensities
    
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

def get_element_data(element, ion_stage='I', wavelength_range=VISIBLE_RANGE):
    """
    Get spectral line data from NIST ASD for any element.
    
    Parameters:
    -----------
    element : str
        Element symbol (e.g., 'H', 'Na', 'O')
    ion_stage : str
        Ionization stage ('I' for neutral, 'II' for singly ionized, etc.)
    wavelength_range : tuple or None
        (min_wavelength, max_wavelength) in Angstroms
        If None, gets all available data
    
    Returns:
    --------
    wavelengths : numpy array
        Ritz wavelengths in Angstroms
    intensities : numpy array
        Relative intensities
    full_table : astropy Table
        Complete data table with all columns
    
    Examples:
    ---------
    # Get hydrogen data in visible range
    wl, intensity, table = get_element_data('H', 'I', (3000, 7000))
    
    # Get all sodium data
    wl, intensity, table = get_element_data('Na', 'I')
    
    # Get oxygen data
    wl, intensity, table = get_element_data('O', 'I', (2000, 10000))
    """

    # Format the spectrum name
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
            return None, None
        
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
        
        return wavelengths, intensities
        
    except Exception as e:
        print(f"Error querying NIST: {e}")
        print("Make sure you have astroquery installed: pip install astroquery")
        return None, None


def print_spectral_lines(wavelengths, intensities, n_lines=20):
    """Print formatted spectral line data"""
    print(f"\n{'Wavelength (Å)':<20} {'Rel. Intensity':<20}")
    print("-" * 40)
    
    # Sort by wavelength
    sorted_idx = np.argsort(wavelengths)
    
    for i in sorted_idx[:n_lines]:
        wl = wavelengths[i]
        intensity = intensities[i]
        
        if np.isnan(intensity):
            intensity_str = "N/A"
        else:
            intensity_str = f"{intensity:.1f}"
        
        print(f"{wl:<20.4f} {intensity_str:<20}")


def get_strongest_lines(wavelengths, intensities, n_strongest=10):
    """
    Get the strongest spectral lines
    
    Returns:
    --------
    strong_wl : numpy array
        Wavelengths of strongest lines
    strong_int : numpy array
        Intensities of strongest lines
    """
    # Filter out NaN intensities
    valid_mask = ~np.isnan(intensities)
    valid_wl = wavelengths[valid_mask]
    valid_int = intensities[valid_mask]
    
    if len(valid_int) == 0:
        print("No intensity data available")
        return np.array([]), np.array([])
    
    # Sort by intensity
    sorted_idx = np.argsort(valid_int)[::-1]  # Descending order
    
    return valid_wl[sorted_idx[:n_strongest]], valid_int[sorted_idx[:n_strongest]]

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

# element -> color and sound data
def get_color_and_sound_data(element):
    """
    Docstring for get_color_and_sound_data
    
    :param element: element symbol
    : output: returns a table of Å, Rel Intensity, Hz, Rel Amplitude, (R, G, B)
    """
    table = []
    wl, ri = get_element_spectrum(element)
    hz = []
    amp = []
    rgb = []

    for l in wl:
        hz.append(A_to_hz_log(l))
    for i in ri:
        amp.append(RI_to_Amp(i, max(ri)))
    for i in range(len(wl)):
        rgb.append(wavelength_to_rgb(wl[i]/10, amp[i]))

    table.append(hz)
    table.append(amp)
    table.append(rgb)
    #print(table)
    for i in range(len(table)):
        print(table[i])

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("NIST ASD SPECTRAL DATA EXTRACTOR")
    print("=" * 70)

    get_color_and_sound_data('H')

    #print(wavelength_to_rgb(600))
    """
    h_wl, h_int, h_table = get_element_data('H', 'I', VISIBLE_RANGE)
    
    if h_wl is not None:
        print_spectral_lines(h_wl, h_int, n_lines=10)
        
        print("\nStrongest lines:")
        strong_wl, strong_int = get_strongest_lines(h_wl, h_int, n_strongest=5)
        print_spectral_lines(strong_wl, strong_int, n_lines=5)
    
    # All oxygen lines in Visible range
    print("\n" + "=" * 70)
    print("Oxygen (O I) - Visible range (4000-7000 Å)")
    print("-" * 70)
    o_wl, o_int, o_table = get_element_data('O', 'I', VISIBLE_RANGE)
    
    if o_wl is not None:
        print_spectral_lines(o_wl, o_int, n_lines=1000)
        
        print("\nStrongest lines:")
        strong_ol, strong_int = get_strongest_lines(o_wl, o_int, n_strongest=50)
        print_spectral_lines(strong_wl, strong_int, n_lines=50)
        """