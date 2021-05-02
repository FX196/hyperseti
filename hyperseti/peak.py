import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi
import time

# Logging
from .log import logger_group, Logger
logger = Logger('hyperseti.peak')
logger_group.add_logger(logger)

construct_max_buffer = cp.RawKernel(r'''
extern "C" __global__
    __global__ void prominent_peaks_kernel
        (const float *img, int M, int N, int min_xdistance, int min_ydistance, float threshold, float *intensity, int *xcoords, int *ycoords, float *max_intensity)
        /* Each thread computes a different dedoppler sum for a given channel

         * img: image, (M x N) shape
         * M: height of img
         * N: width of img
         * min_xdistance: Minimum distance separating features in the x dimension.
         * min_ydistance: Minimum distance separating features in the y dimension.
         * threshold: Minimum intensity of peaks. Default is `0.5 * max(image)`.
         * num_peaks: Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
         *            return `num_peaks` coordinates based on peak intensity.
         * intensity: array of floats that hold the returned peak intensities.
         * xcoords: returned x coordinates of peak intensities.
         * ycoords: returned y coordinates of peak intensities.
         */
        {
            // Setup thread index
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;
            const int p_start_x = tx * min_xdistance;
            const int p_start_y = ty * min_ydistance;
            const int p_end_x = min(N, p_start_x + min_xdistance);
            const int p_end_y = min(M, p_start_y + min_ydistance);
            const int p_mid_y = (p_start_y + p_end_y) / 2;
            // Find local maximum pixel
            float intensity_max = -1.0;
            int x_max = -1;
            int y_max = -1;
            // If block can be divided in half
            int can_divide = p_mid_y > p_start_y && p_mid_y < p_end_y - 1;
            if (can_divide) {
                for (int y = p_start_y; y < p_mid_y; ++y) {
                    for (int x = p_start_x; x < p_end_x; ++x) {
                        // Apply threshold
                        int idx = y * N + x;
                        if (img[idx] > intensity_max) {
                            intensity_max = img[idx];
                            x_max = x;
                            y_max = y;
                        }
                        if (y == p_start_y) {
                            max_intensity[idx] = img[idx];
                        } else {
                            assert(idx - N >= 0);
                            max_intensity[idx] = max_intensity[idx - N] > img[idx] ? max_intensity[idx - N] : img[idx];
                        }
                    }
                }  
                for (int y = p_end_y - 1; y >= p_mid_y; --y) {
                    for (int x = p_start_x; x < p_end_x; ++x) {
                        // Apply threshold
                        int idx = y * N + x;
                        if (img[idx] > intensity_max) {
                            intensity_max = img[idx];
                            x_max = x;
                            y_max = y;
                        }
                        if (y == p_end_y - 1) {
                            max_intensity[idx] = img[idx];
                        } else {
                            assert(idx + N < N * M);
                            max_intensity[idx] = max_intensity[idx + N] > img[idx] ? max_intensity[idx + N] : img[idx];
                            if (y == 202 && x == 1) {
                                printf("hihi %f %f x: %d\n", max_intensity[idx], img[idx], x);
                            }
                        }
                    }
                } 
                if (p_end_y == 208 && p_start_x == 0) {
                    int y = 202;
                    int x = 1;
                    printf("OH NO %f %f %d %d\n", max_intensity[(y-1) * N + x], max_intensity[182 * N + x], p_start_y, p_mid_y);
                }
            } else { // Block cannot be divided in half
                for (int y = p_start_y; y < p_end_y; ++y) {
                    for (int x = p_start_x; x < p_end_x; ++x) {
                        // Apply threshold
                        int idx = y * N + x;
                        if (img[idx] > intensity_max) {
                            intensity_max = img[idx];
                            x_max = x;
                            y_max = y;
                        }
                    }
                } 
            }
        }
''', 'max_buffer_kernel')

prominent_peaks_kernel = cp.RawKernel(r'''
 extern "C" __global__
    __global__ void prominent_peaks_kernel
        (const float *img, int M, int N, int min_xdistance, int min_ydistance, float threshold, float *intensity, int *xcoords, int *ycoords, float *max_intensity)
        /* Each thread computes a different dedoppler sum for a given channel

         * img: image, (M x N) shape
         * M: height of img
         * N: width of img
         * min_xdistance: Minimum distance separating features in the x dimension.
         * min_ydistance: Minimum distance separating features in the y dimension.
         * threshold: Minimum intensity of peaks. Default is `0.5 * max(image)`.
         * num_peaks: Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
         *            return `num_peaks` coordinates based on peak intensity.
         * intensity: array of floats that hold the returned peak intensities.
         * xcoords: returned x coordinates of peak intensities.
         * ycoords: returned y coordinates of peak intensities.
         */
        {
            // Setup thread index
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;
            const int p_start_x = tx * min_xdistance;
            const int p_start_y = ty * min_ydistance;
            const int p_end_x = min(N, p_start_x + min_xdistance);
            const int p_end_y = min(M, p_start_y + min_ydistance);
            const int p_mid_y = (p_start_y + p_end_y) / 2;
            // Find local maximum pixel
            float intensity_max = -1.0;
            int x_max = -1;
            int y_max = -1;
            // If block can be divided in half
            int can_divide = p_mid_y > p_start_y && p_mid_y < p_end_y - 1;
            if (can_divide) {
                for (int y = p_start_y; y < p_mid_y; ++y) {
                    for (int x = p_start_x; x < p_end_x; ++x) {
                        // Apply threshold
                        int idx = y * N + x;
                        if (img[idx] > intensity_max) {
                            intensity_max = img[idx];
                            x_max = x;
                            y_max = y;
                        }
                        if (y == p_start_y) {
                            max_intensity[idx] = img[idx];
                        } else {
                            assert(idx - N >= 0);
                            max_intensity[idx] = max_intensity[idx - N] > img[idx] ? max_intensity[idx - N] : img[idx];
                        }
                    }
                }  
                for (int y = p_end_y - 1; y >= p_mid_y; --y) {
                    for (int x = p_start_x; x < p_end_x; ++x) {
                        // Apply threshold
                        int idx = y * N + x;
                        if (img[idx] > intensity_max) {
                            intensity_max = img[idx];
                            x_max = x;
                            y_max = y;
                        }
                        if (y == p_end_y - 1) {
                            max_intensity[idx] = img[idx];
                        } else {
                            assert(idx + N < N * M);
                            max_intensity[idx] = max_intensity[idx + N] > img[idx] ? max_intensity[idx + N] : img[idx];
                            if (y == 202 && x == 1) {
                                printf("hihi %f %f x: %d\n", max_intensity[idx], img[idx], x);
                            }
                        }
                    }
                } 
                if (p_end_y == 208 && p_start_x == 0) {
                    int y = 202;
                    int x = 1;
                    printf("OH NO %f %f %d %d\n", max_intensity[(y-1) * N + x], max_intensity[182 * N + x], p_start_y, p_mid_y);
                }
            } else { // Block cannot be divided in half
                for (int y = p_start_y; y < p_end_y; ++y) {
                    for (int x = p_start_x; x < p_end_x; ++x) {
                        // Apply threshold
                        int idx = y * N + x;
                        if (img[idx] > intensity_max) {
                            intensity_max = img[idx];
                            x_max = x;
                            y_max = y;
                        }
                    }
                } 
            }
            
            // Barrier, because the next part depends on the previous part
            __syncthreads();

            // Check if local maximum is a peak. Max's are non-inclusive
            int peak_check_min_y = max(0, y_max - min_ydistance);
            int peak_check_max_y = min(M, y_max + min_ydistance + 1);
            int peak_check_min_x = max(0, x_max - min_xdistance);
            int peak_check_max_x = min(N, x_max + min_xdistance + 1);
            int is_peak = 1;

            if (intensity_max != -1.0) {
                // Check elements in the blocks below
                if (p_end_y < M) { // If there are blocks below
                    int start_y_below = p_end_y;
                    int end_y_below = min(M, start_y_below + min_ydistance);
                    int mid_y_below = (start_y_below + end_y_below) / 2;                
                    if (peak_check_max_y - 1 >= mid_y_below && (mid_y_below > start_y_below && mid_y_below < end_y_below - 1)) {
                        // Check against the aggregated max
                        for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                            int idx = (mid_y_below - 1) * N + x;
                            if (max_intensity[idx] > intensity_max) {
                                is_peak = 0;
                                break;
                            }
                        }
                        // Tail case
                        for (int y = mid_y_below; y < peak_check_max_y; ++y) {
                            for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                                if (img[y * N + x] > intensity_max) {
                                    is_peak = 0;
                                    break;
                                }
                            } 
                        }
                    } else if (mid_y_below > start_y_below && mid_y_below < end_y_below - 1) {
                        // Check against aggregated max
                        for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                            int idx = (peak_check_max_y - 1) * N + x;
                            if (max_intensity[idx] > intensity_max) {
                                is_peak = 0;
                                break;
                            }
                        }
                    } else {
                        for (int y = p_end_x; y < peak_check_max_y; ++y) {
                            for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                                if (img[y * N + x] > intensity_max) {
                                    is_peak = 0;
                                    break;
                                }
                            } 
                        }
                    }
                }

                // Check elements in the blocks above
                if (is_peak && ty > 0) {
                    int end_y_above = p_start_y;
                    int start_y_above = max(end_y_above - min_ydistance, 0);
                    int mid_y_above = (start_y_above + end_y_above) / 2;
                    assert(start_y_above < mid_y_above && mid_y_above < end_y_above);
                    assert(end_y_above - min_ydistance == start_y_above);
                    if (peak_check_min_y < mid_y_above) {
                        if (y_max == 208 && x_max == 8) {
                            printf("intensity max %f\n", intensity_max);
                        }
                        // Check against the aggregated max
                        for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                            int idx = mid_y_above * N + x;
                            if (x == 1 && p_start_y == 208) {
                                printf("what %f %f %d\n", max_intensity[182 * N + 1], img[182 * N + 1], mid_y_above);
                            }
                            if (max_intensity[idx] > intensity_max) {
                                is_peak = 0;
                                break;
                            }
                        }
                        // Tail case
                        for (int y = peak_check_min_y; y < mid_y_above; ++y) {
                            for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                                if (img[y * N + x] > intensity_max) {
                                    is_peak = 0;
                                    break;
                                }
                            } 
                        }
                    } else {
                        // Check against aggregated max
                        for (int x = peak_check_min_x; x < peak_check_max_x; ++x) {
                            int idx = peak_check_min_y * N + x;
                            if (max_intensity[idx] > intensity_max) {
                                is_peak = 0;
                                break;
                            }
                        }
                    }
                }

                // Check elements in the left block
                if (is_peak && tx > 0 && can_divide) {
                    for (int x = peak_check_min_x; x < p_start_x; x++) {
                        float top_max = max_intensity[(p_mid_y - 1) * N + x]; // Aggregated max from top
                        float bottom_max = max_intensity[p_mid_y * N + x];
                        if (top_max > intensity_max || bottom_max > intensity_max) {
                            is_peak = 0;
                            break;
                        }
                    }
                } else if (is_peak && tx > 0) {
                    for (int y = p_start_y; y < p_end_y; y++) {
                        for (int x = peak_check_min_x; x < p_start_x; x++) {
                            if (img[y * N + x] > intensity_max) {
                                is_peak = 0;
                                break;
                            }
                        }
                    }
                }

                // Check elements in the right block
                if (is_peak && tx < blockDim.x * gridDim.x - 1 && can_divide) {
                    for (int x = p_end_x; x < peak_check_max_x; x++) {
                        float top_max = max_intensity[(p_mid_y - 1) * N + x]; // Aggregated max from top
                        float bottom_max = max_intensity[p_mid_y * N + x];
                        if (top_max > intensity_max || bottom_max > intensity_max) {
                            is_peak = 0;
                            break;
                        }
                    }
                } else if (is_peak && tx < blockDim.x * gridDim.x - 1) {
                    for (int y = p_start_y; y < p_end_y; y++) {
                        for (int x = p_end_x; x < peak_check_max_x; x++) {
                            if (img[y * N + x] > intensity_max) {
                                is_peak = 0;
                                break;
                            }
                        }
                    }
                }

                if (is_peak) {
                    int t_index = ty * blockDim.x * gridDim.x + tx;
                    intensity[t_index] = intensity_max;
                    xcoords[t_index] = x_max;
                    ycoords[t_index] = y_max;
                }
            }
        }
 ''', 'prominent_peaks_kernel')

def prominent_peaks_optimized(img, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=cp.inf):
    """Return peaks with non-maximum suppression.
    Identifies most prominent features separated by certain distances.
    Non-maximum suppression with different sizes is applied separately
    in the first and second dimension of the image to identify peaks.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_xdistance : int
        Minimum distance separating features in the x dimension.
    min_ydistance : int
        Minimum distance separating features in the y dimension.
    threshold : float
        Minimum intensity of peaks. Default is `0.5 * max(image)`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    intensity, xcoords, ycoords : tuple of array
        Peak intensity values, x and y indices.

    Notes
    -----
    Modified from https://github.com/mritools/cupyimg _prominent_peaks method
    """
    THREADS_PER_BLOCK = (16, 1)
    # min_xdistance, min_ydistance = 2, 2
    # Each thread is responsible for a (min_ydistance * min_xdistance) patch
    # THREADS_PER_BLOCK and img.shape are in the order of (y, x）
    NUM_BLOCKS =  (img.shape[1] // (THREADS_PER_BLOCK[0] * min_xdistance) + int((img.shape[1] % (THREADS_PER_BLOCK[0] * min_xdistance) > 0)),
                   img.shape[0] // (THREADS_PER_BLOCK[1] * min_ydistance) + int((img.shape[0] % (THREADS_PER_BLOCK[1] * min_ydistance) > 0)))
    NUM_THREADS = np.multiply(THREADS_PER_BLOCK, NUM_BLOCKS)
    elems = (NUM_THREADS[0] * NUM_THREADS[1],)
    intensity, xcoords, ycoords, max_intensity = cp.zeros(elems, dtype=cp.float32), cp.zeros(elems, dtype=cp.int32), cp.zeros(elems, dtype=cp.int32), cp.zeros(elems, dtype=cp.float32)
    prominent_peaks_kernel(NUM_BLOCKS, THREADS_PER_BLOCK, (img, cp.int32(img.shape[0]), cp.int32(img.shape[1]), cp.int32(min_xdistance), cp.int32(min_ydistance), cp.float32(threshold), intensity, xcoords, ycoords, max_intensity))
    cp.cuda.Stream.null.synchronize()
    # for i in range(len(intensity)):
    #     if intensity[i] != 0:
    #         print(xcoords[i], ycoords[i], intensity[i], min_xdistance, min_ydistance)
    indices = intensity != 0.0
    return intensity[indices], xcoords[indices], ycoords[indices]

def prominent_peaks(img, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=cp.inf):
    """Return peaks with non-maximum suppression.
    Identifies most prominent features separated by certain distances.
    Non-maximum suppression with different sizes is applied separately
    in the first and second dimension of the image to identify peaks.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_xdistance : int
        Minimum distance separating features in the x dimension.
    min_ydistance : int
        Minimum distance separating features in the y dimension.
    threshold : float
        Minimum intensity of peaks. Default is `0.5 * max(image)`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    intensity, xcoords, ycoords : tuple of array
        Peak intensity values, x and y indices.

    Notes
    -----
    Modified from https://github.com/mritools/cupyimg _prominent_peaks method
    """
    t00 = time.time()
    #img = image.copy()
    rows, cols = img.shape

    if threshold is None:
        threshold = 0.5 * cp.max(img)

    ycoords_size = 2 * min_ydistance + 1
    xcoords_size = 2 * min_xdistance + 1

    t0 = time.time()
    img_max = ndi.maximum_filter(
        img, size=(ycoords_size, xcoords_size), mode="constant", cval=0
    )
    te = (time.time() - t0) * 1e3
    logger.debug(f"Maxfilter: {te:2.2f} ms")

    t0 = time.time()
    mask = img == img_max
    img *= mask
    mask = img > threshold
    te = (time.time() - t0) * 1e3
    logger.debug(f"bitbash: {te:2.2f} ms")

    t0 = time.time()
    # Find array (x,y) indexes corresponding to max pixels
    peak_idxs = cp.argwhere(mask)
    # Find corresponding maximum values
    peak_vals = img[peak_idxs[:, 0], peak_idxs[:, 1]]
    # Sort peak values low to high
    ## Sort the list of peaks by intensity, not left-right, so larger peaks
    ## in Hough space cannot be arbitrarily suppressed by smaller neighbors
    val_sort_idx = cp.argsort(peak_vals)[::-1]
    # Return (x,y) coordinates corresponding to sorted max pixels
    coords = peak_idxs[val_sort_idx]
    te = (time.time() - t0) * 1e3
    logger.debug(f"coord search: {te:2.2f} ms")

    t0 = time.time()
    img_peaks = []
    ycoords_peaks = []
    xcoords_peaks = []

    # relative coordinate grid for local neighbourhood suppression
    ycoords_ext, xcoords_ext = cp.mgrid[
        -min_ydistance : min_ydistance + 1, -min_xdistance : min_xdistance + 1
    ]

    for ycoords_idx, xcoords_idx in coords:
        accum = img_max[ycoords_idx, xcoords_idx]
        if accum > threshold:
            # absolute coordinate grid for local neighbourhood suppression
            ycoords_nh = ycoords_idx + ycoords_ext
            xcoords_nh = xcoords_idx + xcoords_ext

            # no reflection for distance neighbourhood
            ycoords_in = cp.logical_and(ycoords_nh > 0, ycoords_nh < rows)
            ycoords_nh = ycoords_nh[ycoords_in]
            xcoords_nh = xcoords_nh[ycoords_in]

            # reflect xcoords and assume xcoords are continuous,
            # e.g. for angles:
            # (..., 88, 89, -90, -89, ..., 89, -90, -89, ...)
            xcoords_low = xcoords_nh < 0
            ycoords_nh[xcoords_low] = rows - ycoords_nh[xcoords_low]
            xcoords_nh[xcoords_low] += cols
            xcoords_high = xcoords_nh >= cols
            ycoords_nh[xcoords_high] = rows - ycoords_nh[xcoords_high]
            xcoords_nh[xcoords_high] -= cols

            # suppress neighbourhood
            img_max[ycoords_nh, xcoords_nh] = 0

            # add current feature to peaks
            img_peaks.append(accum)
            ycoords_peaks.append(ycoords_idx)
            xcoords_peaks.append(xcoords_idx)

    img_peaks = cp.array(img_peaks)
    ycoords_peaks = cp.array(ycoords_peaks)
    xcoords_peaks = cp.array(xcoords_peaks)
    te = (time.time() - t0) * 1e3
    logger.debug(f"crazyloop: {te:2.2f} ms")

    if num_peaks < len(img_peaks):
        idx_maxsort = cp.argsort(img_peaks)[::-1][:num_peaks]
        img_peaks = img_peaks[idx_maxsort]
        ycoords_peaks = ycoords_peaks[idx_maxsort]
        xcoords_peaks = xcoords_peaks[idx_maxsort]

    te = (time.time() - t0) * 1e3
    logger.debug(f"prominent_peaks total: {te:2.2f} ms")

    return img_peaks, xcoords_peaks, ycoords_peaks