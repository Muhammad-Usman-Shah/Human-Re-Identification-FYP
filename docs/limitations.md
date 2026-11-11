# Limitations

## 1. Limitations stated in the report

Source: report sections 3.5 and 6.6.

- **Same-camera evaluation only.** The system was tested on recorded videos from one camera setup with several viewing angles. It was not tested with several separate cameras or with a live camera feed.
- **Small, self-collected dataset.** There are 14,200 usable crops of more than 30 people. This is enough for project-level fine-tuning but small compared with public re-identification datasets, and the people and recording environment are limited.
- **Small query set.** The query-gallery evaluation uses 35 query images, so the reported mAP and Rank-1 values cannot be generalised widely.
- **Dependence on crop quality.** An incomplete detection box, blur, darkness or partial visibility weakens the OSNet feature and the match.
- **Similar clothing.** Different people in similar clothes can receive close similarity scores, which can lead to wrong merging of identities.
- **Occlusion and tracker ID loss.** Heavy occlusion can cause track loss, a poor crop and a new identity for a returning person.
- **Low light and back views.** Matching accuracy is lower than in normal light and front view.
- **Face cue is often missing.** Faces are not visible in side-view, back-view, low-light or occluded frames, so the face-assisted branch cannot help there.
- **Thresholds were tuned during testing.** Similarity thresholds and margins were set by testing, not by an independent calibration set.
- **Coral TPU covers feature extraction only.** The full YOLOv8 and PyTorch OSNet pipeline was not run on the Coral TPU.

## 2. Limitations of this repository

These points concern what the repository contains, not the research itself.

- **Training and evaluation code is missing.** The notebook loads a trained checkpoint but contains no code for fine-tuning OSNet or for computing mAP, CMC or Rank-1.
- **No weights, dataset or test videos.** `best_model.pth`, the dataset and the input video are not included. The notebook cannot be run as it stands without them.
- **Rank-1 comes from the presentation.** The 97.14% value is on slide 11 of the presentation; it is not stated in the report text and has no script in this repository.
- **Identity overlap is not documented.** It is not stated whether the query and gallery people also appear in the training images.
- **Counting was assessed qualitatively.** No counting error rate against ground-truth counts is reported.
- **Report and notebook differ in some settings.** See section 8 of [methodology.md](methodology.md). The version of the pipeline used for the report's tests appears to differ from the notebook in this repository.
- **Offline processing.** The notebook handles recorded video in two passes. On a Colab GPU its saved output shows about 10 frames per second for the first pass on a 1920 x 1080, 29.99 fps video, which is slower than the video itself.
- **No pinned dependencies.** Package versions were not recorded, and the notebook installs packages at run time.
- **Not re-run.** For this cleanup the notebook code was checked for syntax errors but not executed, because the checkpoint, input video and GPU were not available.
