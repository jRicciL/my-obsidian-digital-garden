---
---

# Biomedical image analysis in python

#images 
#python 
Instructor: Stephen Bailey

## Exploration

### Image Data
- Analyzing tissue composition
- Prediction pathology
- Anatomy reconstruction

#### Toolbox
- ImageIO => #imageio
- NumPy => #numpy
- SciPy => #scipy
- matplotlib => # matplotlib

![[Captura de Pantalla 2021-03-17 a la(s) 13.30.28.png]]

#### Loading images
- `imageio`: read and save images -> including Medical Images
	- `im = imageio.imread(file)` -> `Image` objects
	- The data is loadad as an numpy array
- ==Metadata== -> `imageio` reads metadata into a dictionary.
	- `im.meta` <- python dictionary
- ==Visualization== -> with `maplotlib` -> `imshow`


### N-dimensional images
 #### Images of all shapes and sizes
 
 - Simple **gray-scale images** => `im[row, col]`
 - **Volumetric images** => `im[pln, row, col]`
 - **Colored images** -> Color channels => `im[row, col, chan]`
 - **Movies** -> Time => `im[time, row, col, chan]`

##### Loading volumes directly
```python
import os
import imageio
# Read multiple images from directory
imageio.volread(os.listdir('./chest-data/'))
```


##### Sampling rate and Field of view
- ==Sampling rate== or sampling resolution -> is the amount of space (real space) cover by each pixel (cell in the array) inside the image.
- ==Field of view== -> physical space covered along each axis.
```python
# Compute the field of view
f_view = np.array(vol.meta['sampling']) * vol.shape
```

### Advanced plotting
- Plotting multiple images at once -> `plt.subplots`
- Modifying the aspect ratio:

```python
im = vol[:,:,100]
# Get the sampling of each dimension
d0, d1, d2 = vol.meta['sampling']
# Compute the aspect ratio between two dimensions
asp = d0 / d1
plt.imshow(im, cmap='gray',
		  aspect = asp)
plt.show()
```

![[Captura de Pantalla 2021-03-17 a la(s) 14.08.49.png]]

## Mask and filters
Techniques to emphasize important features in images.

### Image intensities
#### Pixels and voxels
- ==Pixels== -> 2D picture elements
- ==Voxel== -> 3D volume elements

#### Datatypes and image size
![[Captura de Pantalla 2021-03-17 a la(s) 17.02.58.png]]

Check for the size and scale of the image.

```python
import imageio

im = imageio.imread('xray.jpg')
# Check the scale
im.dtype

# Check for the size
im.size
```

#### Histograms
- ==Histograms== -> Summarize the distribution of intensity values in an image.
- `Scipy` -> `scipy.ndimage`
	- Higher-dimensional arrays
	- masked data
- #scikit-image -> Advanced techniques and functionality

##### Histogram

```python
import scipy.ndimage as ndi 

hist = ndi.histogram(im,
		 min = 0,
		 max = 255,
		 bins = 256
		)
```

We will commonly see skewed distributions toward low intensities due to the background.
![[Captura de Pantalla 2021-03-17 a la(s) 17.16.58.png]]

##### Cumulative distribution function

- ==Cumulative distribution function== -> #CDF => Show proportion of pixels that fall within a given range.

In the follow image the ==CDF== shows (in the `y` axis) that almost half of the pixels have values less than 32.

![[Captura de Pantalla 2021-03-17 a la(s) 17.22.50.png]]

##### Histogram Equalization
- ==Equalization== -> redistribute values to optimize full intensity range.
	- Redistribution based on their abundance in the image.

```python
import scipy.ndimage as ndi 

hist = ndi.histogram(...)

# Calculate the CDF
# Compute the rolling sum and divide it by the total
cdf = hist.cumsum() / hist.sum()

# Equalization
im_equalized = cdf[im] * 255 # rescale by 255
```