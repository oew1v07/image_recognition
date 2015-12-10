# im_recog.py setup


Installing Python
=========
* Please download anaconda with Python 3 from: https://www.continuum.io/downloads/
* Install it according to your system using the instructions found on that website
* In your command prompt ensure you have downloaded anaconda by typing
```
conda --version

```
* Also download tqdm (this may require admin rights) by typing
```
pip install tqdm

```

Running im_recog.py
===========

To run the function run_hybrid:
* Load your command prompt (cmd in Windows, terminal in Mac and Linux).
* Change the directory to the one where im_recog.py is located.
* Either make sure your images are also in this folder, or in the command
  give the full filepath as well as the image name.
* Type the following command into the command prompt:
```
ipython
```
* This will load up the ipython program within the command prompt.
* To run any of the code within im_recog.py Type
```
%run im_recog.py
```
  * To run run1 type
 ```
 ma_trs, ma_tsts, n_neighbors, test_out = run1()
 ```
  * To run run2 the training part and the test part have been split so the
    filepaths have to be given. Run the second once the first has completed.
```
run2_train(tr_folder='/Users/user/cw3/training')
```
```
run2_test(test_folder='/Users/user/cw3/training', base_folder='/Users/user/cw3/)
```

  * To run run3 replace the above commands with run3 rather than run2.
  * If using a Windows machine then the file paths should be like the following:

```
'C:\\Users\\user\\cw3'
```
