# from .msg3d_models import msg3d_orig
from . import msg3d, msg3d_bly_fgr
from .eccv_2020_decouple_gcn import decouple_gcn
# from .shift_gcn_2020_cvpr import *
from .iccv_2021_ctr_gcn import ctrgcn, baseline
from .as_gcn_2019_cvpr import as_gcn
from .dual_agcn_2019_cvpr import agcn
from .dgnn_2019_cvpr import dgnn
from .gcn_nas_2020_aaai import agcn3
from .ra_gcn_2019_icip import ra_gcn
from .st_gcn_2018_aaai import st_gcn
from .aaai_2019_motif_stgcn import motif_stgcn
from .qin_2021_simple_st import simple_st
from .qin_2021_component_lab import backbone as component_lab
from .qin_2021_test_model import test_model
from .st_gcn_2018_aaai import st_gcn_koopman
from .st_gcn_2018_aaai import st_gcn_GCL
# Test models
from .model_test_site import frame_idx_regressor
