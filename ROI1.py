import mediapy as media
import cv2
import numpy as np
from skimage import img_as_float
yiq_from_rgb = np.array([[0.299     ,0.587      ,     0.114],
                      [0.59590059,-0.27455667,-0.32134392],
                      [0.21153661, -0.52273617,0.31119955]])
rgb_from_yiq=np.linalg.inv(yiq_from_rgb)
def RGB_to_YIQ(image):
  image=img_as_float(image)         # this coverts all the values of an image to [0,1]
  image=image.dot(yiq_from_rgb.T)   #
  return image
def YIQ_to_RGB(image):
  image=img_as_float(image)
  image=image.dot(rgb_from_yiq.T)
  return image

from imageio import get_reader, get_writer
def load_video1(filename):
    reader = get_reader(filename)
    orig_vid = []
    for i, im in enumerate(reader):
        orig_vid.append(im)
    return np.asarray(orig_vid)
def write_video(location,result,FPS):
  media.write_video('location', result, fps=FPS)
  media.show_video(result)

from skimage.filters import gaussian
def amplitude_weighted_blur(x, weight, sigma):
    if sigma != 0:
        return gaussian(x*weight, sigma, mode="wrap") / gaussian(weight, sigma, mode="wrap")
    return x


def difference_of_iir(delta, rl, rh):

    lowpass_1 = delta[0].copy()
    lowpass_2 = lowpass_1.copy()
    out = zeros(delta.shape, dtype=delta.dtype)
    for i in range(1, delta.shape[0]):
        lowpass_1 = (1-rh)*lowpass_1 + rh*delta[i]
        lowpass_2 = (1-rl)*lowpass_2 + rl*delta[i]
        out[i] = lowpass_1 - lowpass_2
    return out

import math
from numpy import *

def simplify_phase(x):
   #Moves x into the [-pi, pi] range.
    temp= ((x + pi) % (2*pi)) - pi
    return temp

def max_scf_pyr_height(dims):
	return int(log2(min(dims[:2]))) - 2

for i in range(-4,4,1):
  x=simplify_phase(i*pi)
  print(x)
# coverts even multiple of pi into 0 and old into -pi

def get_polar_grid(dims):
    center = ceil((array(dims))/2).astype(int)
    xramp, yramp = meshgrid(linspace(-1, 1, dims[1]+1)[:-1], linspace(-1, 1, dims[0]+1)[:-1])
    theta = arctan2(yramp, xramp)
    r = sqrt(xramp**2 + yramp**2)

    # eliminate the zero at the center
    r[center[0], center[1]] = min((r[center[0], center[1]-1], r[center[0]-1, center[1]]))/2
    return theta,r

def get_angle_mask_smooth(index, num_bands, angle, is_complex):
    order = num_bands-1
    const = sqrt((2**(2*order))*(math.factorial(order)**2)/(num_bands*math.factorial(2*order)))
    angle = simplify_phase(angle+(pi*index/num_bands))

    if is_complex:
        return const*(cos(angle)**order)*(abs(angle) < pi/2)
    else:
        return abs(sqrt(const)*(cos(angle)**order))

def get_filters_smooth_window(dims, orientations, cos_order=6, filters_per_octave=6, is_complex=True, pyr_height=-1):

    max_pyr_height = max_scf_pyr_height(dims)                                   #calculating maxpyramid height that we can take from dim ie. (h,w)
    if( pyr_height == -1 or pyr_height > max_pyr_height):
        pyr_height = max_pyr_height
    total_filter_count = filters_per_octave * pyr_height                        #total fileters is (height of pyramid)*(filters per octave)

    theta, r = get_polar_grid(dims)
    r = (log2(r) + pyr_height)*pi*(0.5 + (total_filter_count / 7)) / pyr_height

    window_function = lambda x, c: (abs(x - c) < pi/2).astype(int)
    compute_shift = lambda k: pi*(k/(cos_order+1)+2/7)

    rad_filters = []

    total = zeros(dims)
    a_constant = sqrt((2**(2*cos_order))*(math.factorial(cos_order)**2)/((cos_order+1)*math.factorial(2*cos_order)))
    for k in range(total_filter_count):
        shift = compute_shift(k+1)
        rad_filters += [a_constant*(cos(r-shift)**cos_order)*window_function(r,shift)]
        total += rad_filters[k]**2
    rad_filters = rad_filters[::-1]

    center = ceil(array(dims)/2).astype(int)
    low_dims = ceil(array(center+1.5)/4).astype(int)
    total_cropped = total[center[0]-low_dims[0]:center[0]+low_dims[0]+1, center[1]-low_dims[1]:center[1]+low_dims[1]+1]

    low_pass = zeros(dims)
    low_pass[center[0]-low_dims[0]:center[0]+low_dims[0]+1, center[1]-low_dims[1]:center[1]+low_dims[1]+1] = abs(sqrt(1+0j-total_cropped))
    total += low_pass**2
    high_pass = abs(sqrt(1+0j-total))

    anglemasks = []
    for i in range(orientations):
        anglemasks += [get_angle_mask_smooth(i, orientations, theta, is_complex)]

    out = [high_pass]
    for i in range(len(rad_filters)):
        for j in range(len(anglemasks)):
            out += [anglemasks[j]*rad_filters[i]]
    out += [low_pass]
    return out

(h,w)=(300,300)
filters=get_filters_smooth_window((h, w), 8, filters_per_octave=2)
len(filters)

def get_radial_mask_pair(r, rad, t_width):
    log_rad = log2(rad)-log2(r)
    hi_mask = abs(cos(log_rad.clip(min=-t_width, max=0)*pi/(2*t_width)))
    lo_mask = sqrt(1-(hi_mask**2))
    return (hi_mask, lo_mask)

def get_angle_mask(b, orientations, angle):
    order = orientations - 1
    a_constant = sqrt((2**(2*order))*(math.factorial(order)**2)/(orientations*math.factorial(2*order)))
    angle2 = simplify_phase(angle - (pi*b/orientations))
    return 2*a_constant*(cos(angle2)**order)*(abs(angle2) < pi/2)

def get_filters(dims, r_vals=None, orientations=4, t_width=1):
    if r_vals is None:
        r_vals = 2**np.array(list(range(0,-max_scf_pyr_height(dims)-1,-1)))
    angle, r = get_polar_grid(dims)
    hi_mask, lo_mask_prev = get_radial_mask_pair(r_vals[0], r, t_width)
    filters = [hi_mask]
    for i in range(1, len(r_vals)):
        hi_mask, lo_mask = get_radial_mask_pair(r_vals[i], r, t_width)
        rad_mask = hi_mask * lo_mask_prev
        for j in range(orientations):
            angle_mask = get_angle_mask(j, orientations, angle)
            filters += [rad_mask*angle_mask/2]
        lo_mask_prev = lo_mask
    filters += [lo_mask_prev]
    return filters

(h,w)=(300,300)
pyr_height = max_scf_pyr_height((h, w))
filters=get_filters((h, w), 2**np.array(list(range(0,-pyr_height-1,-1)), dtype=float), 4)
len(filters)

def phase_amplify1(video, magnification_factor, fl, fh, fs, attenuate_other_frequencies=False, pyramid_type="octave", sigma=0, temporal_filter=difference_of_iir):
    num_frames, h, w, num_channels = video.shape
    pyr_height = max_scf_pyr_height((h, w))

    if pyr_type == "octave":
        print("Using vanilla octave pyramid")
        filters = get_filters((h, w), 2**np.array(list(range(0,-pyr_height-1,-1)), dtype=float), 4)
    elif pyr_type == "halfOctave":
        print("Using half octave pyramid")
        filters = get_filters((h, w), 2**np.array(list(range(0,-pyr_height-1,-1)), dtype=float), 8, t_width=0.75)
    elif pyr_type == "smoothHalfOctave":
        print("Using smooth half octave pyramid.")
        filters = get_filters_smooth_window((h, w), 8, filters_per_octave=2)
    elif pyr_type == "quarterOctave":
        print("Using quarter octave pyramid.")
        filters = get_filters_smooth_window((h, w), 8, filters_per_octave=4)
    else:
        print("Invalid filter type. Specify ocatave, halfOcatave, smoothHalfOctave, or quarterOctave")
        return None
    yiq_video = np.zeros((num_frames, h, w, num_channels))
    fft_video = np.zeros((num_frames, h, w), dtype=complex64)

    for i in range(num_frames):
        yiq_video[i] = RGB_to_YIQ(video[i])
        fft_video[i] = spfft.fftshift(spfft.fft2(yiq_video[i][:,:,0]))

    magnified_y_channel = np.zeros((num_frames, h, w), dtype=complex64)
    dc_frame_index = 0
    for i in range(1,len(filters)-1):
        print("processing level "+str(i))

        dc_frame = spfft.ifft2(spfft.ifftshift(filters[i]*fft_video[dc_frame_index]))
        dc_frame_no_mag = dc_frame / np.abs(dc_frame)
        dc_frame_phase = np.angle(dc_frame)

        total = np.zeros(fft_video.shape, dtype=float)
        filtered = np.zeros(fft_video.shape, dtype=complex64)

        for j in range(num_frames):
            filtered[j] = spfft.ifft2(spfft.ifftshift(filters[i]*fft_video[j]))
            total[j] = simplify_phase(np.angle(filtered[j]) - dc_frame_phase)

        print("bandpassing...")
        total = temporal_filter(total, fl/fs, fh/fs).astype(float)

        for j in range(num_frames):
            phase_of_frame = total[j]
            if sigma != 0:
                phase_of_frame = amplitude_weighted_blur(phase_of_frame, np.abs(filtered[j]), sigma)

            phase_of_frame *= magnification_factor

            if attenuate_other_frequencies:
                temp_orig = np.abs(filtered[j])*dc_frame_no_mag
            else:
                temp_orig = filtered[j]
            magnified_component = 2*filters[i]*spfft.fftshift(spfft.fft2(temp_orig*np.exp(1j*phase_of_frame)))

            magnified_y_channel[j] = magnified_y_channel[j] + magnified_component

    for i in range(num_frames):
            magnified_y_channel[i] = magnified_y_channel[i] + (fft_video[i]*(filters[-1]**2))

    out = np.zeros(yiq_video.shape)

    for i in range(num_frames):
        out_frame  = np.dstack((np.real(spfft.ifft2(spfft.ifftshift(magnified_y_channel[i]))), yiq_video[i,:,:,1:3]))
        out[i] = YIQ_to_RGB(out_frame)

    return out.clip(min=0, max=1)

import cv2
import numpy as np

# Function to process the entire video within the selected region
def process_video1(video_path):
    output_video_path = 'output_video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Read the first frame to get the frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read the first frame.")
        return

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Create an array to store the selected points
    points = []

    # Callback function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Region", frame)

    # Create a window and set the callback function
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", mouse_callback)

    # Display the first frame and wait for user input
    cv2.imshow("Select Region", frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Convert the points to a numpy array
    points = np.array(points, dtype=np.int32)

    # Get the bounding box of the region
    x, y, w, h = cv2.boundingRect(points)

    # Process the entire video
    frames = []
    output_video=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the region defined by the points
        cropped_frame = frame[y:y+h, x:x+w]

        # Append the cropped frame to the list
        frames.append(cropped_frame)

        # Display the result
        cv2.imshow("Cropped Frame", cropped_frame)
        cv2.waitKey(25)  # Adjust the wait key as needed

    cap.release()
    cv2.destroyAllWindows()

    # Combine frames into a single video
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, 25.0, (w, h))

        for frame in frames:
            # Write the cropped frame to the output video
            output_video.write(frame)

        output_video.release()

    media.write_video(output_video_path,output_video)
 
    # return output_video_path

# Load the video
# Replace 'your_video.mp4' with the path to your video file
video_path = 'C:/Users/91934/Desktop/MAV/video1.mp4'

# Process the entire video within the selected region

# process_video1(video_path, output_video_path)

# video_file = "output_video.mp4"
# video = load_video1(video_file)
# video_resized = np.zeros((len(video), 250, 250, 3))
# for i in range(len(video)):
#     video_resized[i] = sktransform.resize(video[i], (250,250))

# magnification_factor = 7
# fl = .04
# fh = .4
# fs = 1
# attenuate_other_frequencies=True
# pyr_type = "octave"
# sigma = 5
# temporal_filter = difference_of_iir
# scale_video = .8

# result1=phase_amplify(video_resized, magnification_factor, fl, fh, fs, attenuate_other_frequencies=attenuate_other_frequencies, pyramid_type=pyr_type, sigma=sigma, temporal_filter=temporal_filter)
