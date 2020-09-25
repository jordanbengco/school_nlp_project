## Disclaimer ##
This project was completed alongside Alex Nguyen. It was re-uploaded from SFU-gitlab to github because the original repository also contained completed homework assignemnts. There are missing files in this project due to file sizes but should work for the most part.

## Contents ##
data/"x": Contains all our data for testing.

yelp_data_extraction.py: Used to create small portions of the original dataset.

word2vec.py: Has the word2vec: continuous bag of words implementation

tfidf.py: Python version of sentiment.ipynb, runs the tf-idf implementation and scores it.

cbow.ipynb: runs the word2vec implementation and scores it.

sentiment.ipynb: runs the tf-idf implementation and scores it.

project.ipynb: Project write-up.

## How to run to check scores ##

Word2vec scores:
Getting scores for word2vec is done in the cbow.ipynb file. Which  scores to calculate depend on the file that is being imported and the test_len variable that is defined. Both of these can be changed in the second cell in the cbow.ipynb file.

To test subset of size 100, 200, 300, 10000.
Change the "test_len" argument in the line "cbow = Word2vec(...)" to 100, 200, 300, 10000 respectively.

Certain test require certain input file. Data file name format will be "reviewX-Y.json". 
For any test that has 100, 200, 300 test size use files with X=500. 
For any test with the test size 10000, use X=20000. 
For test that only includes 1, 3, or 5 star reviews use Y="1-3-5"
For test that only includes 1, or 5 star reviews use Y="1-5"
For test that only includes 2 or 4 star reviews use Y="2-4"
For test that EXCLUDES 3 star reviews use y="3"
For test that includes all reviews use y="all"

For example, to get the score using a test set of 200 and only using 1 or 5 star reviews, change the 'corpus' variable to "data/review500-2-4.json" and the "test_len" argument to 200. Both these changes should be done in the second cell of the cbow.ipynb file.

Tf-idf scores:
Getting scores for the tf-idf model is done in the sentiment.ipynb file. Two varibles that you can change to run different tests are good_stars and sample_size.

To test subset of size 100, 200, 300, 10000:
Change the "sample_size" argument in the second cell to 100, 200, 300, 10000 respectively.

To test different subsets of stars:
Change the "good_stars" variable in the second cell to include which stars you would like to test.

For example, to get the score using a test set of 200 and only using 1 or 5 star reviews, change the sample_size = 200 and good_stars = ['1', '5'].
