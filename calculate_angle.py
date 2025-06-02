import numpy as np

def calculate_angle_3d(a, b, c):
    """
    Calculate the angle at point b formed by 3D points a, b, and c.
    Points a, b, c are numpy arrays with shape (3,).

    Returns:
        angle in degrees (acute angle, always <= 180)
    """
    ba = a - b
    bc = c - b

    # Compute cosine of the angle between vectors ba and bc
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Return the smaller angle (acute)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    if angle_deg > 90:  # optional clamp depending on joint
        angle_deg = 180 - (angle_deg - 90)
        
    return angle_deg


# Optional: quick test
if __name__ == "__main__":
    import numpy as np
    A = np.array([1, 2, 3])
    B = np.array([2, 2, 3])
    C = np.array([3, 4, 3])

    angle = calculate_angle_3d(A, B, C)
    print(f"Angle at B: {angle:.2f} degrees")
