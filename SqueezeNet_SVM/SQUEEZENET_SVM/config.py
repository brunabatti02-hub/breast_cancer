SEED = 42
BASE_PATH = r"C:\Users\franc\OneDrive\Desktop\Breast Cancer\BrestCancer Datasets\BreaKHis"
CSV_PATH = "Folds_SqueezeNet_SVM.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_FOLDS = 5

# MODE: "binary" (benign vs malignant) or "multiclass" (8 subtypes)
MODE = "binary"

# Preprocessing (paper-inspired)
USE_GRAYSCALE = True
APPLY_ADAPTIVE_MEDIAN = True
ADAPTIVE_MEDIAN_MAX = 7
APPLY_ANISOTROPIC_DIFFUSION = True
DIFFUSION_ITER = 10
DIFFUSION_KAPPA = 30
DIFFUSION_GAMMA = 0.1
APPLY_HIST_EQ = True
APPLY_MORPH_OPEN_CLOSE = True
MORPH_KERNEL = 3
APPLY_OTSU_MASK = True
TISSUE_THRESHOLD = 0.6

# Patches (paper uses ROI/patching; for BreakHis we tile)
PATCH_SIZE = 256
PATCH_STRIDE = 256

# Augmentation (paper: rotations, flips, blur, shear, skew)
AUGMENT_TRAIN = True
AUG_PER_IMAGE = 1
AUG_ROTATE = True
AUG_FLIP = True
AUG_SHEAR = 10
AUG_TRANSLATE = 0.05
AUG_BLUR_SIGMAS = (0.25, 2.0)
AUG_SHARPNESS = (0.5, 1.5)
