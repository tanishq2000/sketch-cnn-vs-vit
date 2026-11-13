# sketch-cnn-vs-vit
Should a sketch app ship a CNN or a vision transformer? Frozen ResNet-50 vs DeiT-Small/Base on 20 classes of ImageNet-Sketch, comparing accuracy against latency and memory, plus calibration, failure analysis and confidence-based abstention. Duplicate-aware split, pinned checkpoints, bootstrap CIs.

CNN or Vision Transformer for Sketch Classification 
Should a sketch app ship a CNN or a vision transformer?
Three frozen ImageNet models, measured on accuracy and cost. Nothing is trained. 
Python PyTorch timm Dataset Track 
 Table of contents 
Overview
Models and data
Results
How this is useful
Setup
Reproducibility
Repository structure
Limitations
Data and licences
AI assistance
References 
 Overview 
This project takes three pretrained classifiers from the timm library, runs them unchanged on hand-drawn sketches, and reports:What is measured HowAccuracy Overall, per-class precision, recall and F1, confusion matricesCost Parameter count, milliseconds per image on GPU and CPU, peak GPU memoryConfidence against correctness Reliability diagrams, expected calibration error (ECE), accuracy against coverageFailures The most confident errors and what they have in commonAttention One attention map for the vision transformerMitigation The model abstains (“not sure”) below a confidence threshold 
It was built as Assignment 2 of the Computer Vision course (AIML ZG525) in the M.Tech AI/ML programme at BITS Pilani WILP. 
Models and dataRole Model timm checkpoint ParametersCNN baseline ResNet-50 resnet50.a1_in1k 25.6 MViT baseline DeiT-Small deit_small_patch16_224.fb_in1k 22.1 MIntervention (larger ViT) DeiT-Base deit_base_patch16_224.fb_in1k 86.6 M 
Data: 20 randomly chosen classes of ImageNet-Sketch. That is 1,017 images, of which 804 form the test set. 
ResultsModel Accuracy, 20 classes (95% CI) GPU ms/image CPU ms/image Peak GPU MBResNet-50 65.5% (61.1 to 69.8) 3.18 31.7 115DeiT-Small 68.4% (64.0 to 72.5) 3.13 35.9 89DeiT-Base 66.4% (62.1 to 70.6) 6.44 114.8 339 
Timed at batch size 1 on an NVIDIA RTX 5060 Ti and an Intel i5-14400F. 
Key findings 
A bigger transformer did not help. DeiT-Base costs two to four times more than DeiT-Small and is not more accurate (difference of -2.0 points,
95% CI -5.4 to +1.5).
CNN against ViT is undecided. DeiT-Small is 2.9 points ahead of ResNet-50, but the interval includes zero (-1.0 to +6.7), so the difference is not
established at this sample size.
The dataset contains copies. 443 of the 1,017 images are flipped or rotated copies of another image. A plain random split would have leaked 185 test
images into dev, so the split is made by drawing, not by image.
Confidence is useful but is not a probability. With abstention, DeiT-Small answers 58.8% of test images and is right on 88.2% of those, against
68.4% without abstaining. 
[!NOTE] The full analysis, with all figures, is in CV_Assignment2_Report.pdf. 
How this is useful 
A template for judging pretrained models. To test other models, change the checkpoint ids in the model-loading cell. To test other ImageNet
labelled data, change the dataset cell. The same accuracy, cost and calibration measurements then run on them.
Honest comparison of two models. The bootstrap confidence intervals show whether a gap between two models is real or just noise from a small
test set.Duplicate detection. The split cell finds flipped and rotated copies of the same image and keeps them on one side of the split. This applies to any
scraped image dataset.
Cost measurement done properly. The timing code uses warm-up runs and GPU synchronisation, and measures peak memory per model.
Abstention. The last code cell shows how to turn a confidence score into a “not sure” option, with the threshold chosen on dev and checked once on
test. 
Setup 
Requirements 
Python 3.10 or newer (developed on Python 3.14)
About 10 GB of free disk space (the dataset download is about 7 GB)
A GPU is optional 
[!TIP] Without a GPU the notebook still runs, but it reports model weight size in place of peak GPU memory. 
Installation 
1. Get the code 
git clone https://github.com/<your-username>/sketch-cnn-vs-vit.git
cd sketch-cnn-vs-vit
 2. Create and activate a virtual environment 
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
 3. Install the libraries 
For GPU use, install PyTorch first with the command from pytorch.org. 
pip install torch torchvision timm huggingface_hub numpy scipy scikit-learn pandas matplotlib pillow jupyter
 4. Run the notebook 
jupyter notebook assignment2.ipynb
 Run all cells from top to bottom. The first run downloads the dataset and the three model checkpoints from the Hugging Face Hub and caches them, so
later runs are fast. 
Running on Google Colab 
  Upload assignment2.ipynb to Colab.
Select a GPU runtime (Runtime → Change runtime type).
Uncomment the %pip install line in the first cell.
Run all cells. 
ReproducibilityItem ValueSeed 42, for the choice of classes, the dev/test split and the bootstrapPinned versions The dataset and all three checkpoints are pinned to exact commit hashes in the codeDeterministic Accuracy figures are the same on every runNot deterministic Timings change slightly between runs and depend on the hardware 
Repository structure 
sketch-cnn-vs-vit/
├── assignment2.ipynb├── CV_Assignment2_Report.pdf└── README.md

# the full pipeline, with saved outputs
# four-page report with the analysis and figures
Limitations 
Only 20 classes and 460 unique test drawings, from one random draw of classes.
The models choose among 20 classes, not 1,000, so these accuracies are not comparable with published ImageNet-Sketch numbers.
Timings come from one machine.Data and licences 
ImageNet-Sketch (Wang et al., 2019): the repository is under the MIT licence.
Model checkpoints: from timm, under the Apache-2.0 licence. 
[!IMPORTANT] The sketch images were scraped from the web and copyright stays with the artists. No images are included in this repository.
The notebook downloads them for evaluation only. 
AI assistance 
Claude (Anthropic) suggested the dataset and checkpoints, wrote most of the notebook code and drafted the text. I chose among the options, ran and
checked everything, and reviewed the analysis and conclusions. 
References 
Wang et al., Learning Robust Global Representations by Penalizing Local Predictive Power, NeurIPS 2019
Touvron et al., Training data-efficient image transformers and distillation through attention, ICML 2021
Wightman et al., ResNet strikes back, arXiv:2110.00476, 2021
Geirhos et al., ImageNet-trained CNNs are biased towards texture, ICLR 2019
Naseer et al., Intriguing Properties of Vision Transformers, NeurIPS 2021
