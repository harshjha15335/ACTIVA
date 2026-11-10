# ActionFormer metric provenance
Source: https://github.com/happyharrycn/actionformer_release/blob/61ea7eb9308a568b0cf45e3804830836e30061de/libs/utils/metrics.py
License: MIT, Copyright (c) 2021 University of Wisconsin-Madison.

Only compute_average_precision_detection, segment_iou and interpolated_prec_rec
are retained. Replaced deprecated np.float with float and use stable descending
score ordering (ties preserve input order). Matching, per-threshold GT locks,
precision envelope and AP integration are otherwise retained.
The wrapper resets DataFrame indices and passes one class at a time.
No original GT duplicate-removal or label-ID encoding is applied.
