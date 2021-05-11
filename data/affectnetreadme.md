# AffectNet

AffectNet contains about 1M facial images collected from the Internet by querying three major search engines using 1250 emotion related keywords in six different languages. About half of the retrieved images (~420K) are manually annotated for the presence of seven discrete facial expressions (categorical model) and the intensity of valence and arousal (dimensional model). The rest of the images (~550K) are automatically annotated using ResNext Neural Network trained on all manually annotated training set samples with average accuracy of 65%. 

AffectNet is by far the largest existing database of facial expressions, valence, and arousal in the wild enabling research in automated facial expression recognition in two different emotion models.

AffectNet provides:
- Images of the faces
- Location of the faces in the images
- Location of the 68 facial landmarks
- Eleven emotion and non-emotion categorical labels (Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt, None, Uncertain, No-Face)
- Valence and arousal values of the facial expressions in continuous domain

More details for the labels are provided in the pdf in this directory. 

The data was downloaded from the authors 4/1/2021 and downloaded by Dr. Riley

This document was last updated on 4/1/2021