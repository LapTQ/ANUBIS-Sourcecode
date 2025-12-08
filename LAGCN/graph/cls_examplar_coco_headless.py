import os
import numpy as np


class CLSExamplar(object):
    def __init__(self, topo_str, base_dir="cls_matrix", num_class=None, num_point=None):
        """
        CLSExamplar class for the Anubis dataset.

        Args:
            topo_str: Topology string (used for filenames).
            base_dir: Directory for saving matrix files.
            num_class: Number of classes.
            num_point: Number of joints.
        """
        assert num_class == 14 and num_point == 12, """This code only apply to the following actions and keypoints:
Actions:
    1. standing
    2. standing up
    3. sitting
    4. sitting down
    5. walking
    6. hand slide or stay in body part
    7. hand withdraw from a body part
    8. hand reach out
    9. hand interact ahead
    10. hand pull back
    11. hand withdraw from pocket or bag or basket
    12. hand slide or stay in pants pocket
    13. hand slide or stay in bag
    14. hand slide or stay in basket

Keypoints:
    1. "left_shoulder",
    2. "right_shoulder",
    3. "left_elbow",
    4. "right_elbow",
    5. "left_wrist",
    6. "right_wrist",
    7. "left_hip",
    8. "right_hip",
    9. "left_knee",
    10. "right_knee",
    11. "left_ankle",
    12. "right_ankle",
"""

        # matrix_path = os.path.join(
        #     os.path.dirname(__file__), base_dir, topo_str + ".npy"
        # )

        # # Check for pre-generated matrix.
        # if os.path.exists(matrix_path):
        #     self.A = np.load(matrix_path)
        #     print(f"Loaded exemplar matrix from {matrix_path}")
        # else:
        #     # If the file doesn't exist, create a new matrix.
        #     print(f"Matrix file not found at {matrix_path}, generating new matrix...")
        #     self.A = self.generate_exemplar_matrix(num_class, num_point)

        #     # Create directory and save the matrix.
        #     os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        #     np.save(matrix_path, self.A)
        #     print(f"Saved generated matrix to {matrix_path}")
        self.A = self.generate_exemplar_matrix(num_class, num_point)

    def generate_exemplar_matrix(self, num_class, num_point):
        """
        Generate exemplar matrix.
        """
        # Generate exemplar matrix.
        # Shape: [num_class, num_point, num_point]
        exemplar_matrix = np.zeros((num_class, num_point, num_point), dtype=np.float32)

        # Azure Kinect joint connections.
        inward_ori_index = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (0, 6),
            (1, 7),
            (6, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (9, 11),
        ]

        # Create base adjacency matrix.
        base_adj = np.eye(num_point, dtype=np.float32)
        for i, j in inward_ori_index:
            base_adj[i, j] = 1
            base_adj[j, i] = 1  # Undirected graph

        # Group A: Lower Body / Locomotion (Legs + Hips)
        relevant_lower = [
            (6, 8), (8, 10), # Left Leg
            (7, 9), (9, 11), # Right Leg
            (6, 7)           # Hips
        ]

        # Group B: Upper Body / Interaction (Arms + Shoulders)
        relevant_upper = [
            (0, 2), (2, 4), # Left Arm
            (1, 3), (3, 5), # Right Arm
            (0, 1)          # Shoulders
        ]

        # Generate slightly different exemplar matrices for each class.
        for c in range(num_class):
            # Base: Use normalized adjacency matrix.
            exemplar_matrix[c] = base_adj.copy()

            if 0 <= c <= 4:
                # Classes 0-4: Standing, Sitting, Walking -> Focus on Legs
                current_relevant_pairs = relevant_lower
            else:
                # Classes 5-13: Hand interactions -> Focus on Arms
                current_relevant_pairs = relevant_upper

            for i, j in current_relevant_pairs:
                # We use *= 1.5. Since the value is currently 1.0, it becomes 1.5.
                # If it were 0 (not connected), it stays 0.
                exemplar_matrix[c, i, j] *= 1.5
                exemplar_matrix[c, j, i] *= 1.5

            # Strategy 2: Add some randomness.
            np.random.seed(c)  # Ensure same matrix is generated each time
            noise = np.random.normal(0, 0.01, (num_point, num_point))
            exemplar_matrix[c] += noise * (exemplar_matrix[c] > 0)

            # Ensure diagonal is 1
            np.fill_diagonal(exemplar_matrix[c], 1.0)

            # Symmetrize
            exemplar_matrix[c] = (exemplar_matrix[c] + exemplar_matrix[c].T) / 2

            # Normalize
            D = np.sum(exemplar_matrix[c], axis=1)
            D[D == 0] = 1
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            exemplar_matrix[c] = D_inv_sqrt @ exemplar_matrix[c] @ D_inv_sqrt

        return exemplar_matrix