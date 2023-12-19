import cv2
import numpy as np

def find_xaxis(
    bw_img,
    rho=1,
    theta=np.pi/2,
    thresh=2,
    max_line_gap=3,
    target_cos=1,
    target_frac_len=0.9,
    target_frac_mpos=0.9,
):
    """
    Given a binary array bw_img, return the endpoints
    of the presumed x-axis in flipped array coordinates:
    [n1 m1 n2 m2] (higher n left-to-right, higher m top-to-bottom)

    Uses length, cosine and position to find a long,
    horizontal line close to the bottom of the image.

    Parameters
    ----------
    bw_img: Array
    rho: 
        Distance resolution of the accumulator in pixels.
    theta: 
        Angle resolution of the accumulator in radians.
    thresh: 
        Accumulator threshold parameter. Only those lines 
        are returned that get enough votes (>threshold).
    min_line_length:
        minimum length line to consider (in pixels)
    max_line_gap:
        maximum allowable gap in continguous line (in pixels)
    target_cos: float
        Target cosine value of x-axis line (default 1)
    target_frac_len: float
        Target fractional length of x-axis line, relative to
        width of image (default 0.7)
    target_frac_mpos: float
        Target position of the x-axis line on the "m" axis of
        the image (default 0.8, i.e. "80% of the way to the bottom")
    
    Returns
    -------
    xax_pts: Array (4x1)
        [x1 y1 x2 y2] endpoints of the x-axis line segment
    """
    # Perform Hough Transform
    min_line_length = bw_img.shape[1]/10  # default minimum line length is 10% of image height
    lines = cv2.HoughLinesP(
        bw_img,
        rho=rho,
        theta=theta,
        threshold=thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
        )
    lines_rs = np.reshape(lines, (-1, 4))  # Nx4 array of line endpoints
    # All lines are of form [x1 y1 x2 y2] (i.e. Cartesian with origin at 
    # lower left of image, not array indices)
    
    # Score each candidate line according to target cosine,
    # length and position
    target_feat_vec = np.array(
        [target_cos, target_frac_len, target_frac_mpos]
        )
    line_scores = []
    for line_pts in lines_rs:
        line_vec = line_pts[2:4] - line_pts[0:2]
        line_mag = np.linalg.norm(line_vec, 2)
        line_cos = np.dot(line_vec / line_mag, np.array([1, 0])) ** 2
        line_frac_len = float(np.absolute(line_vec[0])) / bw_img.shape[1]
        line_frac_mpos = float(line_pts[1]) / bw_img.shape[0]
        feat_vec = np.array(
            [line_cos, line_frac_len, line_frac_mpos]
            )
        line_score = np.linalg.norm(target_feat_vec - feat_vec, 2)
        line_scores.append(line_score)
    
    # Return line with lowest distance from target features
    best_line_idx = np.argmin(line_scores)
    xax_pts = lines_rs[best_line_idx, :]
    return xax_pts


def find_yaxis(
    bw_img,
    rho=1,
    theta=np.pi/2,
    thresh=2,
    max_line_gap=3,
    target_cos=0,
    target_frac_len=0.9,
    target_frac_npos=0.1,
):
    """
    Given a binary array bw_img, return the endpoints
    of the presumed y-axis in flipped array coordinates:
    [n1 m1 n2 m2] (higher n left-to-right, higher m top-to-bottom)

    Uses length, cosine and position to find a long,
    vertical line close to the left of the image.

    Parameters
    ----------
    bw_img: Array
    rho: 
        Distance resolution of the accumulator in pixels.
    theta: 
        Angle resolution of the accumulator in radians.
    thresh: 
        Accumulator threshold parameter. Only those lines 
        are returned that get enough votes (>threshold).
    min_line_length:
        minimum length line to consider (in pixels)
    max_line_gap:
        maximum allowable gap in continguous line (in pixels)
    target_cos: float
        Target cosine value of y-axis line (default 1)
    target_frac_len: float
        Target fractional length of y-axis line, relative to
        height of image (default 0.7)
    target_frac_npos: float
        Target position of the y-axis line on the "n" axis of
        the image (default 0.8, i.e. "80% of the way to the right")
    
    Returns
    -------
    yax_pts: Array (4x1)
        [x1 y1 x2 y2] endpoints of the y-axis line segment
    """
    # Perform Hough Transform
    min_line_length = bw_img.shape[1]/10  # default minimum line length is 10% of image height
    lines = cv2.HoughLinesP(
        bw_img,
        rho=rho,
        theta=theta,
        threshold=thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
        )
    lines_rs = np.reshape(lines, (-1, 4))  # Nx4 array of line endpoints
    # All lines are of form [n1 m1 n2 m2] (i.e. like array indices but with
    # N coming firsts
    
    # Score each candidate line according to target cosine,
    # length and position
    target_feat_vec = np.array(
        [target_cos, target_frac_len, target_frac_npos]
        )
    line_scores = []
    for line_pts in lines_rs:
        line_vec = line_pts[2:4] - line_pts[0:2]
        line_mag = np.linalg.norm(line_vec, 2)
        line_cos = np.dot(line_vec / line_mag, np.array([1, 0])) ** 2
        line_frac_len = float(np.absolute(line_vec[0])) / bw_img.shape[0]
        line_frac_npos = float(line_pts[0]) / bw_img.shape[1]
        feat_vec = np.array(
            [line_cos, line_frac_len, line_frac_npos]
            )
        line_score = np.linalg.norm(target_feat_vec - feat_vec, 2)
        line_scores.append(line_score)
    
    # Return line with lowest distance from target features
    best_line_idx = np.argmin(line_scores)
    yax_pts = lines_rs[best_line_idx, :]
    return yax_pts
