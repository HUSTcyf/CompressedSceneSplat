import matplotlib.pyplot as plt

from scipy.signal import medfilt
import numpy as np
import torch

def calculate_stability_metrics(scores, mask_sizes, thresh_range, eval_params=None):
    """
    Calculate stability metrics for both score and mask size across different thresholds.
    
    This function evaluates how stable segmentation masks are to threshold variations by
    measuring the rate of change (gradient) in both relevancy scores and mask sizes.
    Stable segmentations show minimal changes in mask configuration when thresholds are
    slightly adjusted.
    
    Args:
        scores: Array of segmentation confidence scores at different thresholds
        mask_sizes: Array of corresponding mask sizes (as proportion of image) at different thresholds
        thresh_range: Array of threshold values used for evaluation
        eval_params: Dictionary containing parameters for evaluation:
                     - "min_mask_size": Minimum valid mask size as proportion (e.g., 0.00001)
                     - "max_mask_size": Maximum valid mask size as proportion (e.g., 0.95)
    
    Returns:
        Dictionary containing stability metrics:
        - 'smooth_score_grad': Smoothed gradient of scores (lower values indicate more stability)
        - 'smooth_mask_grad': Smoothed gradient of mask sizes (lower values indicate more stability)
        - 'valid_regions': Boolean mask indicating regions where mask size falls within valid range
        
    """
    # Calculate gradients
    score_gradient = np.abs(np.gradient(scores, thresh_range))
    mask_gradient = np.abs(np.gradient(mask_sizes, thresh_range))
    
    # Smooth gradients
    smooth_score_grad = medfilt(score_gradient, kernel_size=5)
    smooth_mask_grad = medfilt(mask_gradient, kernel_size=5)

    # Filter out regions where mask_size > 0.95 and < 0.00001
    valid_regions = (np.array(mask_sizes) > eval_params["min_mask_size"]) & (np.array(mask_sizes) < eval_params["max_mask_size"])
    
    assert len(smooth_score_grad[valid_regions]) != 0, "No valid regions found"

    return {
        'smooth_score_grad': smooth_score_grad,
        'smooth_mask_grad': smooth_mask_grad,
        'valid_regions': valid_regions
    }

def find_stable_regions(stability_metrics, eval_params=None):
    """
    Find continuous regions where both score and mask size gradients are stable.
    
    This function identifies threshold ranges where segmentation results remain
    consistent (stable), which indicates reliable segmentation performance.
    
    Args:
        stability_metrics: Dictionary containing stability metrics:
            - 'smooth_score_grad': Smoothed gradient of scores
            - 'smooth_mask_grad': Smoothed gradient of mask sizes
            - 'valid_regions': Boolean mask of valid regions
        eval_params: Dictionary with evaluation parameters:
            - "stability_thresh": Maximum gradient value considered stable
        min_region_length: Minimum length of a region to be considered stable
    
    Returns:
        List of tuples containing (start_index, end_index) of stable regions
    """

    score_stable = stability_metrics['smooth_score_grad'] < eval_params["stability_thresh"]
    mask_stable = stability_metrics['smooth_mask_grad'] < eval_params["stability_thresh"]
    valid_regions = stability_metrics['valid_regions']
    
    # Both metrics must be stable
    combined_stable = score_stable & mask_stable & valid_regions
    
    # Find continuous stable regions
    stable_regions = []
    start_idx = None
    
    for i in range(len(combined_stable)):
        if combined_stable[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None and i - start_idx >= 5:
            # Region ends, must be at least 5 points long
                stable_regions.append((start_idx, i))
            start_idx = None
    
    # Handle the case where the last region extends to the end
    if start_idx is not None and len(combined_stable) - start_idx >= 5:
        stable_regions.append((start_idx, len(combined_stable)-1))
    
    return stable_regions
    
def compute_dynamic_threshold(valid_map, object_name, eval_params=None, thresh_range=np.arange(0.01, 1, 0.01)):
    
    """
    Computes the optimal threshold for segmentation by analyzing stability across three levels.
    
    This function normalizes outputs from each feature level, evaluates segmentation performance
    across a range of thresholds, identifies stable regions, and selects the feature level and threshold
    that demonstrate the most stable segmentation behavior.
    
    Process:
        1. For each feature level, normalizes the relevancy scores to [0,1]
        2. Evaluates scores and mask sizes at each threshold value
        3. Calculates stability metrics based on how scores and mask sizes change with threshold
        4. Identifies continuous regions where both metrics are stable
        5. For each level, calculates a score sensitivity metric from the stable region
        6. Selects the level with the lowest score sensitivity (most stable)
        7. Returns the chosen level and its optimal threshold value
    """
    n_head = valid_map.shape[0]
    total_pixels = valid_map.shape[1] * valid_map.shape[2]
    score_gradients = []
    thresholds = []
            
    for head_idx in range(n_head):
        output = valid_map[head_idx]

        output = output - torch.min(output)
        output = output / (torch.max(output) -  torch.min(output) + 1e-9)
        output = output.numpy()
        
        # Calculate metrics
        scores = []
        pixel_counts = []
        
        for thresh in thresh_range:
            mask = output > thresh
            score = np.mean(output[mask]) if np.any(mask) else 0
            scores.append(score)
            
            normalized_count = np.sum(mask) / total_pixels
            pixel_counts.append(normalized_count)

        # Calculate stability metrics
        stability = calculate_stability_metrics(scores, pixel_counts, thresh_range, eval_params=eval_params)
        stable_regions = find_stable_regions(stability, eval_params=eval_params)
        
        if len(stable_regions) == 0:
            print(f"Warning: Found {len(stable_regions)} stable regions for {object_name} head {head_idx}")
            score_gradients.append(999)
            thresholds.append(0.5)
        else:
            valid_mask = stability['valid_regions']
            # Find the last stable region
            (start_idx, end_idx) = stable_regions[-1]
            # Find the longest stable region
            # longest_region = max(stable_regions, key=lambda region: region[1] - region[0])
            # (start_idx, end_idx) = longest_region
            if np.any(valid_mask[start_idx:end_idx+1]):
                score_sensitivity = (scores[end_idx]- scores[start_idx]) / (thresh_range[end_idx] - thresh_range[start_idx] + 1e-9)
                score_gradients.append(score_sensitivity)
                thresholds.append((thresh_range[start_idx] + thresh_range[end_idx]) / 2) # take the median threshold
            else:
                score_gradients.append(999)
                thresholds.append(0.5)
                
    chosen_lvl = np.argmin(score_gradients)
    threshold = thresholds[chosen_lvl]
    
    return chosen_lvl, threshold
    

def plot_relevancy_and_threshold(relevancy_map, prompt_name, head_idx, save_path, threshold=0.5):
    """
    Plot relevancy map and thresholded areas side by side
    """
    if torch.is_tensor(relevancy_map):
        relevancy_map = relevancy_map.numpy()
    
    # Create threshold mask
    threshold_mask = relevancy_map > threshold
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot relevancy map
    im1 = ax1.imshow(relevancy_map, cmap='viridis')
    ax1.set_title(f'Relevancy Map\n{prompt_name}, Level {head_idx}')
    fig.colorbar(im1, ax=ax1, label='Relevancy Score')
    ax1.axis('off')
    
    # Plot thresholded map
    im2 = ax2.imshow(threshold_mask, cmap='binary')
    ax2.set_title(f'Thresholded Map (>{threshold})\n{prompt_name}, Level {head_idx}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()