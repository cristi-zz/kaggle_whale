
# My NOAA Whale recognition failure

## tl;dr

Code that detects a whale with 96.81% accuracy.

The code is cutted out from my pipeline. If sth is broken, please comment here or on the forum.

## Technical info about what worked

Histogram and edge statistics + a small classifier detects contours that are whales.

The pipeline starts by detecting some connected regions that are different from the general aspect of the image. Then a classifier decide if these regions are or not whales. If there is only one region it is outputted. If there are more, (or no region predicted by the classifier) the largest one is returned.

I used [anlthms](https://github.com/anlthms/whale-2015)'s bonnet-tip and blowhead json files to train and validate the pipeline.
The training and tuning was done on first 2000 whales from training set. The validation was on the rest of ~2500 images.
96.81% regions had the point1 inside the image and 98.3% had the point2 inside the image. Because point2 is fairly inside the whale, you can say that ~2% of images are not whales.

For how to run the detector look at kaggle_whale\code\src\demo.py -> demo_segment_and_predict_whale()

Because the code is fairly slow (~10 seconds/image) I added a json with all the whales. You have kaggle_whale\images\test_rectangles.json  and kaggle_whale\images\train_rectangles.json . Take a look at kaggle_whale\code\src\demo.py -> demo_load_processed_info_and_cut_out_whale()  on how to use these json files.

When a whale is cutted out from the original image, the destination image has a standard height of 400 pixels and a width 1.3 times longer than the bounding box.

The classifier (dec_tree_model.pkl), some random forest I think is trained on 2000 images. I didn't have time to train it on the whole 4000+ images

Keep reading the code for improvement tips and for how exactly I did it.

For a list of "bad" images check the end of the readme.

## The story (with feelings and pictures)

I started this competition with my mind set on NOT to use CNN's.
So old school techniques. Look at the data, make suppositions, code them and evaluate the results.
I needed an "end-game" solution, to make the final recognition step. I asked some questions on the forums and decided that the best feature is the shape of the white spots. I started to look for a fesable solution (in literature) and voila: [Bag of contour fragments for robust shape classification](http://www.sciencedirect.com/science/article/pii/S0031320313005426) by Wang et al. Cherry on top, they provided the code for their paper! I downloaded the code, ironed some bugs with the help of some examples (replaced the svm and vlfeat) and I was in the game! 

Next step, I took the Vinh Nguyen's annotated whales and started to cook a quick and dirty segmentation. I placed some dots on the spots, and selected the regions that containted those dots. Without much control on the quality I got some results. Some beautiful, some ugly.

The constants in the code, the hard thresholds, the lack of basic intensity invariance were just some elefants that I was trying to ignore. I fed the edges into the BCF code (the Bag of contour fragments paper), used some random forests and I got accuracies close to 50% on ~20 whales! Log loss was about 1.9. All these, without any parameter search except some hand tweaking.

So was I in the game? Apparently yes. I started to work some more on selecting the spots out of the image. I worked out some more convoluted code to get some segmentation out, coupled it with a classifier that predicted if the current segment is or is not a spot. This is where I started to panick and started to skip corners. I worked each step as far as it got me some results, I had no test data, no performance estimation, just visual inspection. So I coded the first steps of the pipeline. Get an image, extract some candidate spots, predict which one are the spots and write out their contours.

The first real test was when I run an evaluation on 180 whales (all these who have more than 10 images/whale) and the results were bad at best.

With the little time that I had, I started to follow my instincts and walked my well worn shoes. Started with a TDD approach, developed each step until a decent detection rate was achieved and all in all, I made significant and accountable progress. See "Technical Info" section about what I could put together.


## What else did I tried

TBA

## What else I would do

TBA

## Lessons (re)learned

Trust your instincts

The shortest path is the known one


## Addendum
"Bad" images taken from the validation set.

    pt1 is out for c:\NOAA\test\w_2_t\w_4326.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4427.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4505.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4529.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_4529.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4620.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_4620.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4661.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_4661.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4797.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_4797.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4883.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_4883.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_4939.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_4939.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5044.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5082.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5084.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5099.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_5099.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5113.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5117.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5137.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5214.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_5214.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5244.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_5244.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5428.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5470.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5554.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_5554.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5773.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_5904.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_5904.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6056.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6161.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6161.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6174.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6174.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6176.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6176.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6252.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6279.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6553.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6606.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6636.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6636.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6669.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6669.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6731.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6731.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6732.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6782.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_6782.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6914.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6939.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_6997.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7025.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7040.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7040.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7109.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7112.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7112.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7268.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7268.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7362.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7362.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7444.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7492.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7531.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7531.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7638.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7725.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7745.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7745.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_7970.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_7970.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8029.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8029.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8152.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8156.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8156.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8257.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8257.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8306.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8382.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8382.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8420.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8420.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8512.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8512.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8567.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8567.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8572.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8587.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8587.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8621.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8750.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8778.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8778.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8869.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8953.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_8991.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_8991.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9036.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9036.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9082.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9095.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9099.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9153.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9153.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9191.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9191.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9249.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9249.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9258.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9275.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9388.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9388.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9402.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9402.jpg
    pt1 is out for c:\NOAA\test\w_2_t\w_9440.jpg
    pt2 is out for c:\NOAA\test\w_2_t\w_9440.jpg