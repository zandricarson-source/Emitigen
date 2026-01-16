"""
SETUP:
Use: wavelengths, intensities, table = get_element_data('H', 'I')
"""

from astroquery.nist import Nist
import astropy.units as u
import numpy as np

VISIBLE_RANGE = (4000, 7000)

def get_element_data(element, ion_stage='I', wavelength_range=None):
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
        
        return wavelengths, intensities, table
        
    except Exception as e:
        print(f"Error querying NIST: {e}")
        print("Make sure you have astroquery installed: pip install astroquery")
        return None, None, None


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


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("NIST ASD SPECTRAL DATA EXTRACTOR")
    print("=" * 70)
    
    # Example 1: Hydrogen in visible range
    print("\nHydrogen (H I) - Visible range (4000-7000 Å)")
    print("-" * 70)
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