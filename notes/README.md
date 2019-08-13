# Notes

## Architecture

#### Friday August, 2, 2019
The categorical variables in the output of the decoder and the predictor always seem to have strange
losses. I would like to try to remove the categorical outputs at least from the predictor.

#### Sunday August 4, 2019
Categorical losses removed from both predictor and decoder.

I am now multiplying the inputs by 10,000. This makes the losses larger than
1, so MSELoss seems to do a bit better. Even with categorical loss in the decoder
this seemed to work pretty well, however the cat loss was around 0.5 but the
decoder loss was around 100 so I dont know how much that loss is actually contributing
to the gradients. Removed all categorical losses for now.
