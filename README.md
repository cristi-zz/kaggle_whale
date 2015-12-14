# kaggle_whale
My NOAA Whale recognition failure


Contents

tl;dr

Technical info about what worked



The story (with feelings and pictures)

I started this competition with my mind set on NOT to use CNN's.
So old school techniques. Look at the data, make suppositions, code them and evaluate the results.
I needed an "end-game" solution, to make the final recognition step. I asked some questions on the forums and decided that the best feature is the shape of the white spots. I started to look for a fesable solution (in literature) and voila: [Bag of contour fragments for robust shape classification](http://www.sciencedirect.com/science/article/pii/S0031320313005426) by Wang et al. Cherry on top, they provided the code for their paper! I downloaded the code, ironed some bugs with the help of some examples (replaced the svm and vlfeat) and I was in the game! 

Next step, I took the Vinh Nguyen's annotated whales and started to cook a quick and dirty segmentation. I placed some dots on the spots, and selected the regions that containted those dots. Without much control on the quality I got some results. Some beautiful, some ugly.

The constants in the code, the hard thresholds, the lack of basic intensity invariance were just some elefants that I was trying to ignore. I fed the edges into the BCF code (the Bag of contour fragments paper), used some random forests and I got accuracies close to 50% on ~20 whales! Log loss was about 1.9. All these, without any parameter search except some hand tweaking.

So was I in the game? Apparently yes. I started to work some more on selecting the spots out of the image. I worked out some more convoluted code to get some segmentation out, coupled it with a classifier that predicted if the current segment is or is not a spot. This is where I started to panick and started to skip corners. I worked each step as far as it got me some results, I had no test data, no performance estimation, just visual inspection. So I coded the first steps of the pipeline. Get an image, extract some candidate spots, predict which one are the spots and write out their contours.

The first real test was when I run an evaluation on 180 whales (all these who have more than 10 images/whale) and the results were bad at best.

With the little time that I had, I started to follow my instincts and walked my well worn shoes. Started with a TDD approach, developed each step until a decent detection rate was achieved and all in all, I made significant and accountable progress. See "Technical Info" section about what I managed to do in little time.


What else did I tried

What else I would do

Lessons learned

