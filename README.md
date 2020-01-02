# Active learning with mismatch-first farthest-traversal
The active learning algorithm of mismatch-first farthest-traversal with visualization

## An imaginary problem for visualization

The visualization uses an binary classification problem, with 5000 randomly generated data points. The data points belong to two classes, visualized with green and blue. The labeling budget is 500. In each batch, 100 data points are queried for labels.

![Image description](http://zsy.fi/static/active/MFFT/figure_0.png)


## Detailed visualization on the model predictions and sample selection
Model predictions             |  Sample selection
:-------------------------:|:-------------------------:
No model available|![a1](http://zsy.fi/static/active/MFFT/figure_a1.png)
![b1](http://zsy.fi/static/active/MFFT/figure_b1.png)| ![a2](http://zsy.fi/static/active/MFFT/figure_a2.png)
![b2](http://zsy.fi/static/active/MFFT/figure_b2.png)| ![a3](http://zsy.fi/static/active/MFFT/figure_a3.png)
![Image description](http://zsy.fi/static/active/MFFT/figure_b3.png)| ![Image description](http://zsy.fi/static/active/MFFT/figure_a4.png)
![Image description](http://zsy.fi/static/active/MFFT/figure_b4.png)| ![Image description](http://zsy.fi/static/active/MFFT/figure_a5.png)
![Image description](http://zsy.fi/static/active/MFFT/figure_b5.png)| ![Image description](http://zsy.fi/static/active/MFFT/figure_a6.png)



## Animated labeling processes along with two other sampling strategies

### Farthest-first traversal
It maximizes the diversity of selected samples. It does not rely on a model, simply select the farthest data point to the traversed set.
![Image description](http://zsy.fi/static/active/FF/FF.gif)

### Uncertainty sampling
It completely relies on a model, selecting the samples with lowest prediction certainty.
![Image description](http://zsy.fi/static/active/Uncertainty/uncertainty.gif)


### Mismatch-first farthest-traversal
It relies on both a existing model and the structure of data points. The primary criterion is prediction mismatch between model-predicted labels and nearset-neighbour predicted labels. The second criterion is the distance to the selected samples.
![Image description](http://zsy.fi/static/active/MFFT/MFFT.gif)

