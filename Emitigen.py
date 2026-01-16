import requests
import pandas as pd
from io import StringIO

def get_sodium_wavelengths(spectrum_type='I', wavelength_range=None):
    """
    Retrieve sodium spectral line data from NIST ASD.
    
    Parameters:
    -----------
    spectrum_type : str
        'I' for neutral sodium (Na I), 'II' for singly ionized (Na II), etc.
    wavelength_range : tuple or None
        (min_wavelength, max_wavelength) in Angstroms, or None for all
    
    Returns:
    --------
    pandas.DataFrame : Spectral line data
    """
    
    # NIST ASD API endpoint
    base_url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
    
    # Parameters for the query
    params = {
        'spectra': f'Na {spectrum_type}',  # Sodium spectrum
        'limits_type': '0',  # No limits initially
        'low_w': '',
        'upp_w': '',
        'unit': '1',  # Angstroms
        'de': '0',  # All energy levels
        'format': '3',  # ASCII output with tabs
        'line_out': '0',  # All lines
        'en_unit': '0',  # cm^-1 for energy
        'output': '0',  # All output columns
        'bibrefs': '1',  # Include bibliographic references
        'page_size': '15',
        'show_obs_wl': '1',  # Show observed wavelengths
        'show_calc_wl': '1',  # Show calculated wavelengths
        'unc_out': '1',  # Show uncertainties
        'order_out': '0',  # Wavelength order
        'max_low_enrg': '',
        'show_av': '2',  # Show transition probabilities
        'max_upp_enrg': '',
        'tsb_value': '0',
        'min_str': '',
        'A_out': '0',  # Transition probabilities
        'intens_out': 'on',  # Intensities
        'max_str': '',
        'allowed_out': '1',
        'forbid_out': '1',
        'min_accur': '',
        'min_intens': '',
        'conf_out': 'on',  # Configuration
        'term_out': 'on',  # Terms
        'enrg_out': 'on',  # Energies
        'J_out': 'on'  # J values
    }
    
    # Add wavelength range if specified
    if wavelength_range:
        params['limits_type'] = '0'
        params['low_w'] = str(wavelength_range[0])
        params['upp_w'] = str(wavelength_range[1])
    
    # Make request
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Parse the response
        text = response.text
        
        # Find the data section (after the header)
        lines = text.split('\n')
        data_start = 0
        
        for i, line in enumerate(lines):
            if '---' in line and i > 0:  # Find separator line
                data_start = i + 1
                break
        
        if data_start > 0:
            # Extract data lines
            data_lines = []
            for line in lines[data_start:]:
                if line.strip() and not line.startswith('='):
                    data_lines.append(line)
            
            # Create DataFrame (simplified parsing)
            print(f"Retrieved {len(data_lines)} spectral lines for Na {spectrum_type}")
            return text  # Return raw text for manual parsing
        else:
            print("No data found in response")
            return None
    else:
        print(f"Error: HTTP {response.status_code}")
        return None


# Example usage
if __name__ == "__main__":
    # Get all neutral sodium (Na I) lines
    print("Fetching sodium emission lines from NIST ASD...")
    data = get_sodium_wavelengths('I')
    
    # Get visible spectrum range (approx 380-750 nm = 3800-7500 Angstroms)
    print("\nFetching visible sodium lines (3800-7500 Å)...")
    visible_data = get_sodium_wavelengths('I', wavelength_range=(3800, 7500))
    
    # Famous sodium D-lines (589-590 nm range)
    print("\nFetching sodium D-lines region (5880-5900 Å)...")
    d_lines = get_sodium_wavelengths('I', wavelength_range=(5880, 5900))
    
    print("\n" + "="*60)
    print("Note: The output is in ASCII format from NIST.")
    print("You may need to parse it further based on your needs.")
    print("="*60)