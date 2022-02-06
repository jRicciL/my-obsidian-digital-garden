---
---

# Image Processing in python

#Course
#python
#images

Instructor: Rebeca Gonzalez

***
---

## Introducing Image Processing and scikit-image

### Numpy for images
Fundamentals of image processing techniques:
- -> Flipping
- -> Extract and analyze features

##### Histogram of an images
Graphical representation of the amount of pixels of each intensity value.
Applications:
- Analysis
- Thresholding
- Brightness and contrast
- Equalize an image

```python
blue_channel = image[:, :, 2]

plt.hist(blue_channel.ravel(), bins = 256)
plt.title('Histogram for the blue channel')
plt.show()
```


### Thresholding
- Simplest method of image segmentation
	- Isolate objects
	- Object detection
- Only form grayscale images
- Categories
	- Global or histogram bases: good for uniform backgrounds
	- Local or adaptative: 

#### Global
```python
from skimage.filters import try_all_threshold
# Obtain all the resulting images
fig, ax = try_all_threshold(image, verbose=False)
```

```python
# Calculate the optimal value
from skimage.filters import threshold_otsu
# Get the optimal 
thr = threshold_otsu(image)
binary_global = image > thr # Apply the thr
```

#### Local
We need to set a block size to apply the local filter
```python
# Import the local threshold
from skimage.filters import threshold_local

# Set.the block size to 35 (pizel regions)
# Local neighbor
block_size = 35
# obtiain the optimal
local_thr = threshols_local(image, block_size, offset=10)
```

## Filtering
- Modify to enhancing an image 
- -> it is a mathematical function that is applied to images (the tensor that represents it):
	- Emphasize or remove features
	- Smoothing
	- Sharpening
	- Edge detection
- It is a neighborhood operation

### Edge detection
Works by detecting discontinuities in brightness.
- Algorithms:
	- #Sobel
	- #Canny


```python
froms skimage.filters import sobel

# Apply tedge detection filter
adge_sobel = sobel(image)
```

![[Captura de Pantalla 2021-01-23 a la(s) 13.06.48.png]]

### Smoothing
#### Gaussian Smoothing
Used to reduce noise or blur an image.
-> Gaussian filter blurs edges and reduce contrast:

```python
from skimage.filters import gaussian

# Apply the filter
gaussian_image = gaussian(image,
		  multichanel=True) # True if the image is colored
```

### Contrast enhancement
- Often used for medical images
- The contrast of an images can be seen as ==the ratio of its dynamic range== -> **The spread of the histogram**

![[Captura de Pantalla 2021-01-23 a la(s) 13.17.16.png]]

##### [[Contrast stretching]]
It stretch the histogram to the full range of possible values.

##### [[Histogram equalization]]
Spreads out the most frequent histogram intensity values using probability distributions.

Three types:
1. Standard
2. Adaptive Histogram equalization:
	- Works over image tiles or neighbors
	- Computes several histograms, each corresponding to a distinct part of the image and use them to redistribute the lightness values of the image histogram.
1. The Limited Adaptive (**CLAHE**):
	- Developed to prevent over-amplification of noise -> It is more natural

```python
from skimage import exposure

# Normal histogram equalization
image_eq = exposure.equalize_hist(image)

# CLAHE
image_adapeq = exposure.equalize_adapthist(image,
		# Clip limit, ranges from 0 to 1
		# Regularizes the image contrast
		 clip_limit=0.03)

```

### Transformations
##### Why transform images?
- Preparing images for classification #ML models
- Optimization and compression of images
- Save images with same proportion

#### Rotating and Rescaling
```python
from skimage.transform import rotate, rescale

# ROTATE
image_rotated = rotate(image, -90) #clockwise (to the right)

# RESCALE
image_rescaled = rescale(
		image,
		1/4,
		anti_aliasing=True, # smooth
		multichannel=True
	)
```

#### Aliasing
It is makes the image look like it has waves or ripples radiating from a certain portion

#### Resize
- It let us to specify a specific size instead of a scaling proportion.
```python
from skimage.transfom import rezise
h = 400
w = 500

# Resize it
image_resized = resize(image,
			(h, w),
			anti_aliasing=True
		)
```

### Morphology
Identify shapes within an image.

##### Shapes
```python
from skimage import morphology
square = morphology.square(4) 
# It is something like create a squared matrix
```

#### Morphological filtering
- Applied over binary images, although can be extended to grayscale ones.

Morphological operations:
- Dilation: 
	- Add pixels to the boundaries of the shapes of the image
	- Expands objects in an images

```python
from skimage import morphology

dilated_image = morphology.binary_dilation(image)
```

- Erosion:
	- Removes pixels from object boundaries.

```python
from skimage import morphology

# Set structuring element to the rectangular-shaped
selem = rectangle(12, 6)

# Obtain the erosed image with binary erosion 
# using a rectangle to perform the erosion
eroded_image = morphology.binary_erosion(image, selem=selem)

# The default shape to erode is a cross
```


## Image Restoration
#### Image reconstruction
- Fixing damaged images
- Text removing
- Logo removing
- Object removing

##### Inpainting
- Reconstructing lost parts of images
- Looking at the non-damaged regions

```python
from skimage.restoration import inpaint

# Obtain the mask
mask = get_mask(defect_image)

# Apply inpainting to damaged image using the mask
restored_image = inpaint.inpaint_biharmonic(defect_image, mask, multichannel=True)
```

## Noise
- Departures from an ideal signal in a image => color grains

### Add noise
```python
from skimage.util import random_noise

# Salt and peper noice
noisy_image = random_noise(image)
```

### Reduce noise
#### Denoising types
- Total variation filter (TV):
	- Cartoon-like images
- Bilateral:
	- Smooths images while persirving images
	- Replace the intensity of each pixel with a weighted average of intensity values form nearby pixels
- Wavelet denoising
- Non-local means denoising

##### Total Variation Filter & bilateral ==Chambolle method==
```python
from skimage.restoration \
	import denoise_tv_chambolle, denoise_bilateral

denoised_image = denoise_tv_chambolle(noisy_image,
						weight=0.1,
						multichannel=True)

# Bilateral
denoised_image = denoise_tv_chambolle(noisy_image,
						multichannel=True)
```


### ==Superpixels== and ==Segmentation==
- Partition images into regions and sections
- Thresholding is the simplest method of segmentation

- ==Superpixels==: Groups of connected pixels with similar colors or gray levels.
- Benefits:
	- more meaningful regions
	- Computational efficiency

- ==Segmentation types==:
	- Supervised -> 
	- Unsupervised ->

#### Simple Linear Iterative Clustering (SLIC)
- A segmentation technique based on superpixels
- Segments the image using k-means clustering

```python
# Import the modules
from skimage.segmentation import slic
from skimage.color import label2rgb
# Obtain the segments
segments = segmentation.slic(image, n_segments=300)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, image, kind='avg')
```


### Finding ==Contours==

A contour is a closer shape of points or line segments representing the outlines of objects.
Uses:
- Measure size
- Classify shapes
- Determine the number of objects in an image

-> The input should be a **binary image**:
- Transform to gray scale
- Apply threshold

```python
from skimage import measure
from skimage.filters import threshold_otsu
from skimage import color

image = color.rgb2gray(image)

# obtain the thresh value
thresh = threshold_otsu(image)

# Applying thresholding
thresholding_image = image > thresh

# Find countours
contours = measure.find_contours(thresholded_image, 0.8)

# contours is a list
```

![[Captura de Pantalla 2021-02-22 a la(s) 18.57.17.png]]

## Advanced operations, detecting faces and features

### Finding the edges with ==Canny==
#Canny

==Canny edge detection==: Widely considered to be the standard edge detection method in image processing.  
-> Produces higher accuracy detecting edges and less execution time compared with #Sobel

```python
from skimage.feature import canny

# Requires a grayscale image
coins = color.rgb2gray(image)

# Apply Canny detector
canny_edges = canny(coins,
				# intensity of the gaussian filter
				   sigma=0.5)
```

- The lower the value of `sigma` the less filter effect applied on the image.


### ==Corner== Detection

Why it is useful?
- Motion detection
- Image registration
- Video tracking
- 3d modelling
- Panorama stitching
- Object recognition

##### Points of interest
Are points in the image which are 
- invariant to rotation
- translation
- intensity 
- scale changes

##### Corners are points of interest
- ==Harris== Corner detector (#harris_corner_detector)

```python
from skimage.feature import corner_harris, corner_peaks

# convert to grayscale
image = rgb2gray(image)

# apply the harris corner detector
meausure_image = corner_harris(image)

# find the corners in the measure_image
coords = corner_peaks(measure\_image,
					 min_distance=5)
```

##### Show image with contours

```python
def show_image_with_corners(image, coords, title="Corners detected"):
	plt.imshow(image, interpolation='nearest', cmap='gray')
	plt.title(title)
	plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
	plt.axis('off')
	plt.show()
```


### ==Face== detection

#### Detecting faces with scikit-image

-> We use a Cascade of ==Classifiers==
- `Cascade` => Creates a window that rolls over the image until find something similar to a human face

```python
from skimage.feature import Cascade

# Load the trainded file from the module root
trained_file = data.lbp_frontal_face_cascade_filename()

# Initializa the detector cascade
detector = Cacade(trained_file)

# Apply the detector over the target image
detected = detector.detect_multi_scale(
	img=image,
	scale_factor=1.2,
	step_ratio=1, # exhaustive search
	min_size=(10, 5),
	max_size=(200, 200)
)

# will return a dictionary
```

### Real World applications

#### Privacy protection

Detect face and anonimate them

```python
# Detect the facs
detected = detector.detect_multiple_scale(
	img=image,
	scle_factor=1.2,
	step_ratio=1,
	min_size=(50,50),
	max_size=(100,100)
)
```
