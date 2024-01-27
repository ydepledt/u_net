import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np

# Some useful colors
COLORS = {'BLUE': '#3D6FFF',
          'RED': '#FF3D3D',
          'ORANGE': '#FF8E35',
          'PURPLE': '#BB58FF',
          'GREEN': '#32CD32',
          'YELLOW': '#F9DB00',
          'PINK': '#FFC0CB',
          'BROWN': '#8B4513',
          'CYAN': '#00FFFF',
}

def compute_number_of_cells(image_path: str,
                            threshold_cells_inf: float = None,
                            threshold_cells_sup: float = None,
                            visualize_hist: bool = True,
                            verbose: bool = True) -> int:
    """
    Compute the number of cells in an image.

    Parameters
    ----------
    image_path : str
        Path to the image.
    
    threshold_cells_inf : float, optional
        Lower threshold for cells. If None, no lower threshold is applied.

    threshold_cells_sup : float, optional
        Upper threshold for cells. If None, no upper threshold is applied.
    
    visualize_hist : bool, optional
        Whether to visualize the histogram of the image.
    
    verbose : bool, optional
        Whether to print information about the image.

    Returns
    -------
    n_labels : int
        Number of cells in the image.
    
    Examples
    --------
    >>> compute_number_of_cells('data/image_1.png', threshold_cells_inf=0.2, threshold_cells_sup=0.8)
    """

    image = plt.imread(image_path)
    
    # Scale the image between 0 and 1
    image = image / 255.0

    shape_before = image.shape
    # Slicing to reshape the image
    image = image[:, :, 0]

    # The image is now a 2D array, which is easier to work with
    if verbose:
        print(f"Shape before: {shape_before}")
        print(f"Shape after:  {image.shape}")

    if visualize_hist:
        image_2 = np.copy(image)

        # Histogram of the image
        plt.hist(image_2.ravel(), bins=255, color=COLORS['BLUE'])

        plt.grid()
        plt.show()

    # Here bacteria is a binary mask of the image (so it's a 2D array of True/False values)
    if threshold_cells_sup and threshold_cells_inf:
        bacteria_mask = (image > threshold_cells_inf) & (image < threshold_cells_sup)
    elif threshold_cells_sup:
        bacteria_mask = image < threshold_cells_sup
    elif threshold_cells_inf:
        bacteria_mask = image > threshold_cells_inf
    else:
        print("You must specify at least one threshold.")

    bacteria = ndi.binary_opening(bacteria_mask, iterations=4)

    labels, n_labels = ndi.label(bacteria)
    print(f"Number of labels: {n_labels}")

    plt.imshow(labels, cmap='rainbow')