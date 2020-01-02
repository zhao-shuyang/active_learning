# Active learning with mismatch-first farthest-traversal
The active learning algorithm of mismatch-first farthest-traversal with visualization

## An imaginary problem for visualization

The visualization uses an imaginary binary classification problem and randomly generated data points. There are 5000 data points belonging to two classes, visualized with green and blue. The labeling budget is 500. In each batch, 100 data points are queried for labels.

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
![Image description](http://zsy.fi/static/active/FF/FF.gif)

### Uncertainty sampling
![Image description](http://zsy.fi/static/active/Uncertainty/uncertainty.gif)


### Mismatch-first farthest-traversal
![Image description](http://zsy.fi/static/active/MFFT/MFFT.gif)

