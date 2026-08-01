# Model card — genre_classifier

## Model details

- **Name:** genre_classifier  
- **Type:** CNN (Conv2D ×3 + Dense) on MFCC inputs  
- **Framework:** Keras 3 / TensorFlow  
- **Files:** `models/genre_classifier.keras`, `models/genre_mapping.json`  
- **Input:** `(126, 13, 1)` float32  
- **Output:** 10-class softmax  

## Intended use

- Educational demos, portfolio, offline experimentation on short clips resembling GTZAN conditions.

## Out of scope

- Commercial genre tagging at scale  
- Fine-grained subgenres, mood, language, or cultural identification  
- Safety-critical decisions  

## Training data

- **GTZAN** research dataset (10 genres × ~100 tracks × 30 s).  
- Known issues: limited diversity, label noise, historical collection biases.  

## Metrics

- Original Colab training used ~30 epochs, 80/20 split (not stratified in notebook).  
- Package `train.py` uses **stratified** split + early stopping; re-run training to populate `train_metrics.json`.  

## Ethical considerations

- Genre labels are Western-pop taxonomies and can misrepresent hybrid or non-Western music.  
- Do not present predictions as objective cultural truth.  

## Caveats

- Not SOTA.  
- Mapping must stay synchronized with weights.  
