import numpy as np


class CLSExamplar(object):
    def __init__(self, num_class=None, num_point=None):
        """
        Class-specific exemplar topology for the 10-class fs26 setup (v259+).

        Args:
            num_class: Number of classes (must be 10).
            num_point: Number of joints (must be 12, coco_headless).
        """
        assert num_class == 10 and num_point == 12, """This code only apply to the following actions and keypoints:
Actions:
    0. cho_tay_vao_tui_quan          (hand in pants pocket)
    1. cho_tay_vao_tui_ao_tui_xach   (hand in shirt pocket / bag)
    2. cho_tay_vao_gio_xe_hang       (hand in basket / cart)
    3. cho_tay_vao_ke                (hand reach shelf)
    4. rut_tay_ve                    (hand withdraw)
    5. dung_yen                      (standing)
    6. ngoi_yen                      (sitting)
    7. dung_day                      (standing up)
    8. ngoi_xuong                    (sitting down)
    9. di                            (walking)

Keypoints:
    0. left_shoulder,  1. right_shoulder,
    2. left_elbow,     3. right_elbow,
    4. left_wrist,     5. right_wrist,
    6. left_hip,       7. right_hip,
    8. left_knee,      9. right_knee,
    10. left_ankle,    11. right_ankle
"""
        self.A = self.generate_exemplar_matrix(num_class, num_point)

    def generate_exemplar_matrix(self, num_class, num_point):
        # Shape: [num_class, num_point, num_point]
        exemplar_matrix = np.zeros((num_class, num_point, num_point), dtype=np.float32)

        inward_ori_index = [
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (0, 6),
            (1, 7), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
        ]

        base_adj = np.eye(num_point, dtype=np.float32)
        for i, j in inward_ori_index:
            base_adj[i, j] = 1
            base_adj[j, i] = 1

        L_SHOULDER, R_SHOULDER = 0, 1
        L_ELBOW, R_ELBOW       = 2, 3
        L_WRIST, R_WRIST       = 4, 5
        L_HIP, R_HIP           = 6, 7
        L_KNEE, R_KNEE         = 8, 9
        L_ANKLE, R_ANKLE       = 10, 11

        pairs_legs = [
            (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
            (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
            (L_HIP, R_HIP),
        ]
        pairs_arms = [
            (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
            (L_SHOULDER, R_SHOULDER),
        ]
        pairs_wrist_hip = [(L_WRIST, L_HIP), (R_WRIST, R_HIP)]
        pairs_elbow_hip = [(L_ELBOW, L_HIP), (R_ELBOW, R_HIP)]
        pairs_wrist_shoulder = [(L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)]

        class_pairs = {
            # hand in pants pocket -> arms + strong wrist/hip
            0: pairs_arms + pairs_wrist_hip,
            # hand in shirt pocket / bag -> complex interaction
            1: pairs_arms + pairs_wrist_hip + pairs_elbow_hip + pairs_wrist_shoulder,
            # hand in basket / cart -> complex interaction
            2: pairs_arms + pairs_wrist_hip + pairs_elbow_hip + pairs_wrist_shoulder,
            # hand reach shelf -> arms
            3: pairs_arms,
            # hand withdraw -> arms + wrist/hip
            4: pairs_arms + pairs_wrist_hip,
            # standing / sitting / standing up / sitting down / walking -> legs
            5: pairs_legs,
            6: pairs_legs,
            7: pairs_legs,
            8: pairs_legs,
            9: pairs_legs,
        }

        for c in range(num_class):
            exemplar_matrix[c] = base_adj.copy()

            for i, j in class_pairs[c]:
                if exemplar_matrix[c, i, j] == 0:
                    # non-physical semantic link
                    exemplar_matrix[c, i, j] = 1.5
                    exemplar_matrix[c, j, i] = 1.5
                else:
                    # physical connection, amplify
                    exemplar_matrix[c, i, j] *= 1.5
                    exemplar_matrix[c, j, i] *= 1.5

            np.random.seed(c)  # deterministic
            noise = np.random.normal(0, 0.01, (num_point, num_point))
            exemplar_matrix[c] += noise * (exemplar_matrix[c] > 0)

            np.fill_diagonal(exemplar_matrix[c], 1.0)
            exemplar_matrix[c] = (exemplar_matrix[c] + exemplar_matrix[c].T) / 2

            D = np.sum(exemplar_matrix[c], axis=1)
            D[D == 0] = 1
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            exemplar_matrix[c] = D_inv_sqrt @ exemplar_matrix[c] @ D_inv_sqrt

        return exemplar_matrix
